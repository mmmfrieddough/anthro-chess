import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from anthro_chess import __version__
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import AcquisitionResult, PreparationResult
from anthro_chess.evaluation import MoveValidationMetrics
from anthro_chess.interfaces.cli import main
from anthro_chess.training import TrainingConfig, TrainingResult


def test_smoke_command_needs_no_external_resources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke"]) == 0
    assert capsys.readouterr().out == (
        f"Anthro Chess {__version__} is installed and ready.\n"
    )


def test_command_results_stay_on_stdout_while_logs_use_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--log-level", "DEBUG", "smoke"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"Anthro Chess {__version__} is installed and ready.\n"
    assert "Starting command smoke" in captured.err
    assert "Completed command smoke" in captured.err


def test_help_only_advertises_implemented_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "smoke" in help_text
    assert "data" in help_text
    assert "train" in help_text
    assert "eval" in help_text
    for planned_command in ("play", "uci"):
        assert f"  {planned_command} " not in help_text


def test_eval_help_advertises_the_puzzle_rating_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["eval", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "prepare-puzzles" in output
    assert "puzzles" in output


def test_data_prepare_command_routes_to_importable_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = Path(__file__).parents[2]
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"
    config = repository_root / "configs/data/lichess-sample.toml"
    output = tmp_path / "artifacts"

    assert (
        main(["data", "prepare", str(sample), str(output), "--config", str(config)])
        == 0
    )

    assert pq.read_table(output / "normalized/games.parquet").num_rows == 1
    command_output = capsys.readouterr().out
    assert "Prepared 1 game(s); rejected 0." in command_output
    assert "manifests/manifest.json" in command_output


def test_eval_freeze_command_routes_to_importable_pool_builder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = Path(__file__).parents[2]
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"
    data_config = repository_root / "configs/data/lichess-sample.toml"
    corpus = tmp_path / "corpus"
    assert (
        main(
            [
                "data",
                "prepare",
                str(sample),
                str(corpus),
                "--config",
                str(data_config),
                "--set",
                "split.validation_fraction=0.1",
                "--set",
                "split.test_fraction=0.85",
            ]
        )
        == 0
    )
    capsys.readouterr()

    pool_config = tmp_path / "pool.toml"
    pool_config.write_text(
        'pool_id = "cli-fixture"\n'
        f'normalized = "{corpus / "normalized"}"\n'
        f'manifest = "{corpus / "manifests/manifest.json"}"\n'
        'split = "test"\n'
    )
    output = tmp_path / "pool"

    assert main(["eval", "freeze", str(output), "--config", str(pool_config)]) == 0

    assert pq.read_table(output / "games.parquet").num_rows == 1
    command_output = capsys.readouterr().out
    assert "Froze 1 game(s)" in command_output
    assert "Identity:" in command_output


def test_eval_freeze_reports_a_configuration_error_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool_config = tmp_path / "pool.toml"
    pool_config.write_text(
        'pool_id = "missing"\n'
        f'normalized = "{tmp_path / "absent"}"\n'
        f'manifest = "{tmp_path / "absent.json"}"\n'
    )

    assert (
        main(["eval", "freeze", str(tmp_path / "pool"), "--config", str(pool_config)])
        == 2
    )

    assert "anthro eval freeze:" in capsys.readouterr().err


def test_eval_freeze_uses_data_root_for_checked_in_artifact_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"
    data_config = repository_root / "configs/data/lichess-sample.toml"
    data_root = tmp_path / "datasets"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))
    assert (
        main(
            [
                "data",
                "prepare",
                str(sample),
                "--config",
                str(data_config),
                "--set",
                "split.validation_fraction=0.1",
                "--set",
                "split.test_fraction=0.85",
            ]
        )
        == 0
    )
    capsys.readouterr()

    pool_config = tmp_path / "pool.toml"
    pool_config.write_text(
        'pool_id = "rooted-fixture"\n'
        'normalized = "artifacts/lichess-sample/normalized"\n'
        'manifest = "artifacts/lichess-sample/manifests/manifest.json"\n'
    )

    assert main(["eval", "freeze", "--config", str(pool_config)]) == 0

    assert pq.read_table(data_root / "rooted-fixture/games.parquet").num_rows == 1


def test_data_prepare_uses_data_root_when_output_is_omitted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"
    config = repository_root / "configs/data/lichess-sample.toml"
    data_root = tmp_path / "datasets"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))

    assert main(["data", "prepare", str(sample), "--config", str(config)]) == 0

    output = data_root / "lichess-sample"
    assert pq.read_table(output / "normalized/games.parquet").num_rows == 1
    assert "Prepared 1 game(s); rejected 0." in capsys.readouterr().out


def test_data_acquire_command_routes_to_importable_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-sample.toml"
    archive_path = tmp_path / "raw/archive.pgn.zst"
    monkeypatch.setattr(
        "anthro_chess.data.acquire_archive",
        lambda output, resolved: AcquisitionResult(
            archive_path=archive_path,
            sha256="a" * 64,
            size_bytes=123,
            reused=False,
        ),
    )

    assert main(["data", "acquire", str(tmp_path), "--config", str(config)]) == 0

    command_output = capsys.readouterr().out
    assert f"Acquired verified archive: {archive_path}" in command_output
    assert f"SHA-256: {'a' * 64}" in command_output


def test_data_acquire_uses_data_root_when_output_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-sample.toml"
    data_root = tmp_path / "datasets"
    captured_output: list[Path] = []
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))

    def fake_acquire(output: Path, _resolved: object) -> AcquisitionResult:
        captured_output.append(output)
        return AcquisitionResult(
            archive_path=output / "raw/archive.pgn.zst",
            sha256="a" * 64,
            size_bytes=123,
            reused=False,
        )

    monkeypatch.setattr("anthro_chess.data.acquire_archive", fake_acquire)

    assert main(["data", "acquire", "--config", str(config)]) == 0
    assert captured_output == [data_root / "lichess-sample"]


def test_data_prepare_infers_shared_archive_independently_of_prepared_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-blitz-2017-04.toml"
    data_root = tmp_path / "datasets"
    captured_paths: list[tuple[Path, Path]] = []
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))

    def fake_prepare(
        input_path: Path,
        output: Path,
        _resolved: object,
    ) -> PreparationResult:
        captured_paths.append((input_path, output))
        return PreparationResult(
            normalized_paths=(output / "normalized/games.parquet",),
            manifest_path=output / "manifests/manifest.json",
            accepted_games=1,
            rejected_games=0,
            split_counts={"train": 1, "validation": 0},
        )

    monkeypatch.setattr("anthro_chess.data.prepare_pgn", fake_prepare)

    assert (
        main(
            [
                "data",
                "prepare",
                "--config",
                str(config),
                "--set",
                'artifact_name="proof-slice"',
            ]
        )
        == 0
    )
    assert captured_paths == [
        (
            data_root / "lichess-blitz-2017-04/raw/"
            "lichess_db_standard_rated_2017-04.pgn.zst",
            data_root / "proof-slice",
        )
    ]


def test_train_command_routes_to_importable_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "training.toml"
    config.write_text(
        """
[train]
normalized = "normalized"
manifest = "manifest.json"

[train.loader]
split = "train"
""",
        encoding="utf-8",
    )
    run_path = tmp_path / "run.json"
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "checkpoints/step-00000002.pt"
    validation = MoveValidationMetrics(
        position_count=1,
        move_loss=7.0,
        legal_move_loss=2.5,
        mask_penalty=4.5,
        legal_mass=0.01,
        top1_illegal_rate=1.0,
        uniform_over_legal_move_loss=3.0,
        uniform_over_vocabulary_move_loss=7.5,
        rated_position_count=1,
        missing_rating_position_count=0,
        missing_rating_move_loss=None,
        rating_slices=(),
    )

    def fake_run(
        resolved: ResolvedConfig[TrainingConfig],
        *,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        assert resolved.value.steps == 2
        return TrainingResult(
            run_path=run_path,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            steps=2,
            initial_parameter_sha256="a",
            final_parameter_sha256="b",
            validation=validation,
        )

    monkeypatch.setattr("anthro_chess.training.run_training", fake_run)

    assert main(["train", "--config", str(config), "--set", "steps=2"]) == 0

    command_output = capsys.readouterr().out
    assert "Completed 2 optimizer step(s)." in command_output
    assert f"Run: {run_path}" in command_output
    assert f"Metrics: {metrics_path}" in command_output
    assert f"Checkpoint: {checkpoint_path}" in command_output
    assert (
        "Validation: raw_move_loss=7.000000 legal_move_loss=2.500000 "
        "uniform_over_legal=3.000000"
    ) in command_output


def test_train_uses_machine_roots_for_checked_in_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "training.toml"
    config.write_text(
        """
output_directory = "artifacts/example-run"

[train]
normalized = "artifacts/example-data/normalized"
manifest = "artifacts/example-data/manifests/manifest.json"

[train.loader]
split = "train"
""",
        encoding="utf-8",
    )
    data_root = tmp_path / "datasets"
    run_root = tmp_path / "runs"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(run_root))

    def fake_run(
        resolved: ResolvedConfig[TrainingConfig],
        *,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        assert resolved.value.output_directory == run_root / "example-run"
        assert resolved.value.train.normalized == data_root / "example-data/normalized"
        assert (
            resolved.value.train.manifest
            == data_root / "example-data/manifests/manifest.json"
        )
        return TrainingResult(
            run_path=run_root / "example-run/run.json",
            metrics_path=run_root / "example-run/metrics.jsonl",
            checkpoint_path=run_root / "example-run/checkpoints/step-00000001.pt",
            steps=1,
            initial_parameter_sha256="a",
            final_parameter_sha256="b",
            validation=None,
        )

    monkeypatch.setattr("anthro_chess.training.run_training", fake_run)

    assert main(["train", "--config", str(config)]) == 0


def test_train_explicit_path_overrides_win_over_machine_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "training.toml"
    config.write_text(
        """
[train]
normalized = "artifacts/default/normalized"
manifest = "artifacts/default/manifests/manifest.json"

[train.loader]
split = "train"
""",
        encoding="utf-8",
    )
    explicit_output = tmp_path / "explicit-run"
    explicit_normalized = tmp_path / "explicit-data/normalized"
    explicit_manifest = tmp_path / "explicit-data/manifests/manifest.json"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(tmp_path / "runs"))

    def fake_run(
        resolved: ResolvedConfig[TrainingConfig],
        *,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        assert resolved.value.output_directory == explicit_output
        assert resolved.value.train.normalized == explicit_normalized
        assert resolved.value.train.manifest == explicit_manifest
        return TrainingResult(
            run_path=explicit_output / "run.json",
            metrics_path=explicit_output / "metrics.jsonl",
            checkpoint_path=explicit_output / "checkpoints/step-00000001.pt",
            steps=1,
            initial_parameter_sha256="a",
            final_parameter_sha256="b",
            validation=None,
        )

    monkeypatch.setattr("anthro_chess.training.run_training", fake_run)

    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--set",
                f'output_directory="{explicit_output}"',
                "--set",
                f'train.normalized="{explicit_normalized}"',
                "--set",
                f'train.manifest="{explicit_manifest}"',
            ]
        )
        == 0
    )


def _record_fixture_results(store_root: Path) -> None:
    """Record two comparable results so the report command has history."""

    from datetime import UTC, datetime

    from anthro_chess.evaluation.results import (
        BenchmarkReference,
        CheckpointReference,
        ResultsStore,
        build_result,
        dataset_reference,
        measurement,
        projection_content_digest,
    )
    from anthro_chess.evaluation.results.metrics import MOVE_PREDICTION_PROJECTION

    rows = [
        {
            "game_id": game_id,
            "ruleset": "standard",
            "initial_position": "startpos",
            "action_ids": [1, 2, 3],
            "white_normalized_rating": 1500,
            "black_normalized_rating": 1500,
        }
        for game_id in (1, 2)
    ]
    component = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    store = ResultsStore(store_root)
    for label, value, day in (("checkpoint-a", 3.5, 1), ("checkpoint-b", 3.2, 8)):
        store.append(
            build_result(
                kind="held-out-prediction",
                benchmark=BenchmarkReference(name="move-validation", version=1),
                checkpoint=CheckpointReference(label=label),
                data=dataset_reference(
                    pool_id="fixture-pool",
                    pool_version=1,
                    view="canonical",
                    selected_games=component.games,
                    game_ids_sha256="a" * 64,
                    components=[component],
                ),
                measurements=[measurement("held_out.move_loss", value, data=component)],
                recorded_at=datetime(2026, 7, day, tzinfo=UTC),
            )
        )


def test_eval_report_shows_the_compact_delta_view(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_fixture_results(tmp_path / "results")

    assert main(["eval", "report", "--store", str(tmp_path / "results")]) == 0

    output = capsys.readouterr().out
    assert "checkpoint-b" in output
    assert "held_out.move_loss" in output
    assert "better" in output


def test_eval_report_emits_machine_readable_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_fixture_results(tmp_path / "results")

    assert (
        main(
            [
                "eval",
                "report",
                "--store",
                str(tmp_path / "results"),
                "--format",
                "json",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["current"]["label"] == "checkpoint-b"
    assert report["baseline"]["label"] == "checkpoint-a"
    # Rows hang off the series they were measured on rather than off the family
    # directly, so a benchmark writing one result per matrix cell cannot have
    # its cells collapse into one row.
    assert any(
        group["metrics"] for family in report["families"] for group in family["series"]
    )


def test_eval_report_resolves_its_store_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_fixture_results(tmp_path / "results")
    monkeypatch.setenv("ANTHRO_CHESS_RESULTS_ROOT", str(tmp_path / "results"))

    assert main(["eval", "report", "--history", "held_out.move_loss"]) == 0

    output = capsys.readouterr().out
    assert "held_out.move_loss" in output
    assert "checkpoint-a" in output


def test_eval_report_reports_an_unknown_selection_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_fixture_results(tmp_path / "results")

    assert (
        main(
            [
                "eval",
                "report",
                "--store",
                str(tmp_path / "results"),
                "--current",
                "checkpoint-z",
            ]
        )
        == 2
    )
    assert "anthro eval report:" in capsys.readouterr().err


def test_eval_tensorboard_projects_the_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "results"
    output = tmp_path / "tensorboard-history"
    _record_fixture_results(store)

    assert (
        main(
            [
                "eval",
                "tensorboard",
                str(output),
                "--store",
                str(store),
            ]
        )
        == 0
    )

    assert len(tuple(output.rglob("events.out.tfevents.*"))) == 1
    rendered = capsys.readouterr().out
    assert "2 points" in rendered
    assert "2 checkpoints" in rendered


def test_eval_metrics_lists_the_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["eval", "metrics", "--format", "json"]) == 0

    registry = json.loads(capsys.readouterr().out)
    families = {family["identifier"] for family in registry["families"]}
    assert {"training-health", "legality", "rating-behavior", "generated-play"} <= (
        families
    )
    efficiency = next(
        family
        for family in registry["families"]
        if family["identifier"] == "training-efficiency"
    )
    assert efficiency["metrics"]
    assert all(metric["execution_sensitive"] for metric in efficiency["metrics"])


def test_eval_budget_reports_quality_against_the_training_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_budget_store(tmp_path / "results")
    monkeypatch.setenv("ANTHRO_CHESS_RESULTS_ROOT", str(tmp_path / "results"))

    assert main(["eval", "budget", "--positions", "2000", "--format", "json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["metric"] == "held_out.move_loss"
    assert [point["processed_positions"] for point in report["points"]] == [1000, 2000]
    assert report["answers"][0]["point"]["checkpoint"] == "run-step-00000200"


def test_eval_budget_reports_a_join_it_cannot_make_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHRO_CHESS_RESULTS_ROOT", str(tmp_path / "empty"))

    assert main(["eval", "budget"]) == 2

    assert "anthro eval budget:" in capsys.readouterr().err


def _write_budget_store(root: Path) -> None:
    """Record two checkpoints that each carry a budget point and a quality one."""

    from datetime import UTC, datetime

    from anthro_chess.evaluation.results import (
        BenchmarkReference,
        CheckpointReference,
        ResultsStore,
        build_result,
        dataset_reference,
        execution_reference,
        measurement,
        projection_content_digest,
    )
    from anthro_chess.evaluation.results.metrics import MOVE_PREDICTION_PROJECTION

    rows = [
        {
            "game_id": game_id,
            "ruleset": "standard",
            "initial_position": "startpos",
            "action_ids": [1, 2, 3],
            "white_normalized_rating": 1500,
            "black_normalized_rating": 1500,
        }
        for game_id in (1, 2)
    ]
    component = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    execution = execution_reference(
        device="cpu",
        device_name="fixture-cpu",
        precision="float32",
        torch_version="2.7.0",
        platform_key="Linux-x86_64",
        platform="Linux-6.1-x86_64",
        workload={"benchmark_version": 1},
    )
    store = ResultsStore(root)
    for step, positions, seconds, loss in (
        (100, 1000, 10.0, 4.0),
        (200, 2000, 20.0, 3.1),
    ):
        label = f"run-step-{step:08d}"
        recorded_at = datetime(2026, 7, 30, 12, step // 100, tzinfo=UTC)
        store.append(
            build_result(
                kind="training-efficiency",
                benchmark=BenchmarkReference(name="training-efficiency", version=1),
                checkpoint=CheckpointReference(label=label, step=step),
                execution=execution,
                measurements=[
                    measurement(
                        "training.processed_positions",
                        float(positions),
                        workload=execution.workload_component(),
                    ),
                    measurement(
                        "training.training_seconds",
                        seconds,
                        workload=execution.workload_component(),
                    ),
                ],
                recorded_at=recorded_at,
            )
        )
        store.append(
            build_result(
                kind="held-out-preview",
                benchmark=BenchmarkReference(name="held-out-preview", version=1),
                checkpoint=CheckpointReference(label=label, step=step),
                data=dataset_reference(
                    pool_id="fixture-pool",
                    pool_version=1,
                    view="preview",
                    selected_games=component.games,
                    game_ids_sha256="a" * 64,
                    components=[component],
                ),
                measurements=[measurement("held_out.move_loss", loss, data=component)],
                recorded_at=recorded_at,
            )
        )


def test_eval_decisions_reads_a_stored_payload_and_writes_the_detail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import chess

    from anthro_chess.chess import encode_move
    from anthro_chess.evaluation.games import (
        GAME_RECORD_VERSION,
        DecisionPolicy,
        DecisionRecord,
        GameOutcome,
        GameTermination,
        SeatRecord,
        build_game_record,
    )

    action_ids = tuple(
        encode_move(chess.Move.from_uci(uci)) for uci in ("e2e4", "e7e5")
    )
    seat = SeatRecord(
        kind="model",
        label="model-a",
        seed=1,
        configuration={"target_rating": 1500, "temperature": 1.0},
    )
    record = build_game_record(
        initial_position=chess.STARTING_FEN,
        prefix_plies=0,
        action_ids=action_ids,
        white=seat,
        black=seat,
        seed=2,
        decisions=[
            DecisionRecord(
                ply_index=0,
                slot="white",
                action_id=action_ids[0],
                policy=DecisionPolicy(
                    enabled_action_count=20,
                    selected_probability=0.4,
                    selected_rank=1,
                    preferred_action_id=action_ids[0],
                    preferred_probability=0.4,
                ),
            ),
            DecisionRecord(
                ply_index=1,
                slot="black",
                action_id=action_ids[1],
                policy=DecisionPolicy(
                    enabled_action_count=20,
                    selected_probability=0.1,
                    selected_rank=4,
                    preferred_action_id=action_ids[0],
                    preferred_probability=0.5,
                ),
            ),
        ],
        outcome=GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=True,
        ),
    )
    games = tmp_path / "games.json"
    games.write_text(
        json.dumps({"version": GAME_RECORD_VERSION, "games": [record.as_record()]}),
        encoding="utf-8",
    )
    output = tmp_path / "detail" / "decisions.json"

    assert (
        main(["eval", "decisions", "--games", str(games), "--output", str(output)]) == 0
    )

    printed = capsys.readouterr().out
    assert "Decisions classified: 2" in printed
    assert "model preference" in printed
    assert "sampling" in printed
    detail = json.loads(output.read_text(encoding="utf-8"))
    assert detail["overall"]["departures"] == 1
    assert [entry["selected_rank"] for entry in detail["samples"]] == [1, 4]


def test_eval_rollout_reports_a_configuration_error_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The prefix arm without a pool is a configuration error, not a skip."""

    config = tmp_path / "rollout.toml"
    config.write_text('arms = ["human-prefix"]\n', encoding="utf-8")

    assert main(["eval", "rollout", "--config", str(config), "--no-record"]) == 2

    assert "anthro eval rollout:" in capsys.readouterr().err


def test_eval_rollout_roots_its_checked_in_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped selection names its pool the way every artifact path is named."""

    from anthro_chess.evaluation import RolloutBenchmarkConfig
    from anthro_chess.interfaces.cli import _resolve_rollout_roots

    data_root = tmp_path / "datasets"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))
    resolved = ResolvedConfig(
        value=RolloutBenchmarkConfig.model_validate(
            {"pool": "artifacts/example-pool", "reference": {"enabled": False}}
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )

    rooted = _resolve_rollout_roots(resolved, [])

    assert rooted.value.pool == data_root / "example-pool"
    # An explicit override is the caller's own path and is left alone.
    overridden = _resolve_rollout_roots(resolved, ["pool=/somewhere/else"])
    assert overridden.value.pool == Path("artifacts/example-pool")


def test_eval_rollout_leaves_a_pool_free_suite_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suite that reads no pool has nothing to root."""

    from anthro_chess.evaluation import RolloutBenchmarkConfig
    from anthro_chess.interfaces.cli import _resolve_rollout_roots

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    resolved = ResolvedConfig(
        value=RolloutBenchmarkConfig.model_validate({"reference": {"enabled": False}}),
        provenance=ConfigProvenance(source=None, overrides=()),
    )

    assert _resolve_rollout_roots(resolved, []).value.pool is None


def test_eval_rollout_renders_every_cell_with_its_series(tmp_path: Path) -> None:
    """The text view has to name each cell's series and what it played."""

    import torch

    from anthro_chess.chess import ACTION_VOCABULARY_SIZE
    from anthro_chess.data import DecisionContext
    from anthro_chess.evaluation import RolloutBenchmarkConfig, benchmark_rollout
    from anthro_chess.evaluation.results import CheckpointReference
    from anthro_chess.interfaces.cli import _render_rollout

    class Runner:
        def predict(self, context: DecisionContext) -> torch.Tensor:
            generator = torch.Generator().manual_seed(len(context.plies))
            return torch.randn(ACTION_VOCABULARY_SIZE, generator=generator)

    resolved = ResolvedConfig(
        value=RolloutBenchmarkConfig.model_validate(
            {
                # The renderer's cell section is what this covers, so the
                # comparison stays off rather than dragging a pool in.
                "reference": {"enabled": False},
                "grid": {
                    "target_ratings": (1200, 1800),
                    "temperatures": (1.0,),
                    "seeds": (0,),
                },
                "generation": {
                    "games_per_position": 1,
                    "maximum_generated_plies": 4,
                    "swap_colors": False,
                },
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )
    result = benchmark_rollout(
        resolved,
        runner=Runner(),
        checkpoint=CheckpointReference(label="fixture-checkpoint", step=1),
    )

    rendered = _render_rollout(result)

    assert "Games: 2 across 2 matrix cell(s)" in rendered
    assert "standard-start rating=1200 temperature=1" in rendered
    assert "standard-start rating=1800 temperature=1" in rendered
    assert "series workload" in rendered
    # One game per cell, cut off at the four-ply limit this fixture declares.
    assert rendered.count("unfinished     1 at the ply limit") == 2
    # Repertoire, waypoint rate, and book depth are separate lines because they
    # answer separate questions: which opening, how far, and how far it could
    # have gone.
    assert "repertoire     " in rendered
    assert "waypoints      " in rendered
    assert "book depth     " in rendered
    assert "available plies" in rendered
    assert "Recorded: nothing" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 120


def test_eval_ladder_reports_a_configuration_error_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reference temperature off the grid is a configuration error."""

    config = tmp_path / "ladder.toml"
    config.write_text(
        "[grid]\ntemperatures = [0.5, 1.0]\nreference_temperature = 0.7\n",
        encoding="utf-8",
    )

    assert main(["eval", "ladder", "--config", str(config), "--no-record"]) == 2

    assert "anthro eval ladder:" in capsys.readouterr().err


def test_eval_ladder_roots_its_checked_in_opening_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped selection names its pool the way every artifact path is named."""

    from anthro_chess.evaluation import LadderBenchmarkConfig
    from anthro_chess.interfaces.cli import _resolve_ladder_roots

    data_root = tmp_path / "datasets"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))
    resolved = ResolvedConfig(
        value=LadderBenchmarkConfig.model_validate(
            {"openings": {"pool": "artifacts/example-pool"}}
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )

    rooted = _resolve_ladder_roots(resolved, [])

    assert rooted.value.openings.pool == data_root / "example-pool"
    # An explicit override is the caller's own path and is left alone.
    overridden = _resolve_ladder_roots(resolved, ["openings.pool=/somewhere/else"])
    assert overridden.value.openings.pool == Path("artifacts/example-pool")


def test_eval_ladder_renders_the_transfer_function_and_its_error_profile(
    tmp_path: Path,
) -> None:
    """The text view has to show ordering, shape, and the profile beside it."""

    import torch

    from anthro_chess.chess import ACTION_VOCABULARY_SIZE, RESIGNATION_ACTION_ID
    from anthro_chess.data import DecisionContext
    from anthro_chess.evaluation import LadderBenchmarkConfig, benchmark_ladder
    from anthro_chess.evaluation.results import CheckpointReference
    from anthro_chess.interfaces.cli import _render_ladder

    class Runner:
        def predict(self, context: DecisionContext) -> torch.Tensor:
            rating = context.target_rating or 1000
            logits = torch.full((ACTION_VOCABULARY_SIZE,), (rating - 2000) / 500.0)
            logits[RESIGNATION_ACTION_ID] = 0.0
            return logits

    resolved = ResolvedConfig(
        value=LadderBenchmarkConfig.model_validate(
            {
                "runtime": {"resignation_enabled": True},
                "grid": {
                    "target_ratings": (1200, 2000),
                    "temperatures": (1.0,),
                    "reference_temperature": 1.0,
                    "seeds": (0, 1),
                },
                "generation": {
                    "games_per_position": 4,
                    "maximum_generated_plies": 20,
                },
            }
        ),
        provenance=ConfigProvenance(source=None, overrides=()),
    )
    result = benchmark_ladder(
        resolved,
        runner=Runner(),
        checkpoint=CheckpointReference(label="fixture-checkpoint", step=1),
    )

    rendered = _render_ladder(result)

    assert "Ladder at temperature=1" in rendered
    assert "configured" in rendered
    assert "ordering       " in rendered
    assert "transfer       slope" in rendered
    # Strength and error profile share one table on purpose: a temperature that
    # holds the score rate while moving the profile has changed the shape of
    # the mistakes, and neither column shows that alone.
    assert "preferred" in rendered
    assert "1200@t1" in rendered
    assert "ablated@t1" in rendered
    assert "Recorded: nothing" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 120


def test_eval_decisions_reports_an_unreadable_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    games = tmp_path / "games.json"
    games.write_text("{", encoding="utf-8")

    assert main(["eval", "decisions", "--games", str(games)]) == 2
    assert "anthro eval decisions:" in capsys.readouterr().err


def test_eval_bridge_records_lists_and_revokes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = str(tmp_path / "results")

    assert (
        main(
            [
                "eval",
                "bridge",
                "add",
                "--store",
                store,
                "--from",
                "a" * 64,
                "--to",
                "b" * 64,
                "--reason",
                "storage format change only",
                "--author",
                "maintainer",
            ]
        )
        == 0
    )
    recorded = capsys.readouterr().out
    bridge_id = recorded.splitlines()[0].removeprefix("Recorded bridge ")

    assert main(["eval", "bridge", "list", "--store", store]) == 0
    assert bridge_id in capsys.readouterr().out

    assert main(["eval", "bridge", "revoke", "--store", store, bridge_id]) == 0
    capsys.readouterr()
    assert main(["eval", "bridge", "list", "--store", store]) == 0
    assert "No bridges are recorded." in capsys.readouterr().out


def _record_sampling_floor(store_root: Path, *, floor: float, games: int) -> None:
    """Record a data-sampling floor of the kind an evaluation run bootstraps."""

    from datetime import UTC, datetime

    from anthro_chess.evaluation.results import (
        FloorEntry,
        ResultsStore,
        build_characterization,
        projection_content_digest,
        series_fingerprint,
    )
    from anthro_chess.evaluation.results.metrics import MOVE_PREDICTION_PROJECTION

    rows = [
        {
            "game_id": game_id,
            "ruleset": "standard",
            "initial_position": "startpos",
            "action_ids": [1, 2, 3],
            "white_normalized_rating": 1500,
            "black_normalized_rating": 1500,
        }
        for game_id in (1, 2)
    ]
    component = projection_content_digest(rows, MOVE_PREDICTION_PROJECTION)
    ResultsStore(store_root).append_characterization(
        build_characterization(
            kind="data-sampling",
            method="bootstrap-over-games",
            replicates=1_000,
            source="the fixture pool",
            floors=[
                FloorEntry(
                    metric="held_out.move_loss",
                    fingerprint=series_fingerprint("held_out.move_loss", component),
                    floor=floor,
                    dispersion=floor / 2.0,
                    sampling_units=games,
                )
            ],
            recorded_at=datetime(2026, 7, 9, tzinfo=UTC),
        )
    )


def test_eval_noise_characterizes_training_noise_from_replicate_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = str(tmp_path / "results")
    _record_fixture_results(tmp_path / "results")

    assert (
        main(
            [
                "eval",
                "noise",
                "characterize",
                "--store",
                store,
                "--kind",
                "training",
                "--checkpoint",
                "checkpoint-a",
                "--checkpoint",
                "checkpoint-b",
                "--metric",
                "held_out.move_loss",
                "--source",
                "two smoke-scale seeds",
            ]
        )
        == 0
    )
    assert "held_out.move_loss" in capsys.readouterr().out

    assert main(["eval", "noise", "list", "--store", store]) == 0
    listed = capsys.readouterr().out
    assert "training" in listed
    assert "two smoke-scale seeds" in listed

    # The report now judges the delta against the floor it just characterized
    # rather than reporting that no floor is known.
    assert main(["eval", "report", "--store", store]) == 0
    assert "(training)" in capsys.readouterr().out


def test_eval_noise_characterize_needs_more_than_one_replicate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_fixture_results(tmp_path / "results")

    assert (
        main(
            [
                "eval",
                "noise",
                "characterize",
                "--store",
                str(tmp_path / "results"),
                "--kind",
                "training",
                "--checkpoint",
                "checkpoint-a",
                "--source",
                "one run",
            ]
        )
        == 2
    )
    assert "at least two checkpoints" in capsys.readouterr().err


def test_eval_noise_plan_sizes_an_axis_from_a_measured_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_sampling_floor(tmp_path / "results", floor=0.04, games=1_000)

    assert (
        main(
            [
                "eval",
                "noise",
                "plan",
                "--store",
                str(tmp_path / "results"),
                "--metric",
                "held_out.move_loss",
                "--effect",
                "0.01",
                "--format",
                "json",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["required_games"] == 16_000
    assert plan["measured_games"] == 1_000


def test_eval_noise_plan_reports_a_missing_floor_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_fixture_results(tmp_path / "results")

    assert (
        main(
            [
                "eval",
                "noise",
                "plan",
                "--store",
                str(tmp_path / "results"),
                "--metric",
                "held_out.move_loss",
                "--effect",
                "0.01",
            ]
        )
        == 2
    )
    assert "no data-sampling floor is recorded" in capsys.readouterr().err


def test_eval_noise_list_says_when_nothing_is_characterized(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["eval", "noise", "list", "--store", str(tmp_path / "results")]) == 0

    assert "No noise characterization is recorded." in capsys.readouterr().out


def _inference_config(path: Path, checkpoint: Path) -> Path:
    """Write the smallest inference-benchmark selection that still measures."""

    path.write_text(
        "\n".join(
            [
                "[model]",
                f'checkpoint_path = "{checkpoint}"',
                'device = "cpu"',
                "",
                "[runtime]",
                "seed = 0",
                "",
                "[latency]",
                "reference_plies = 4",
                "sweep_plies = [4]",
                "decisions = 2",
                "warmup_decisions = 0",
                "",
                "[throughput]",
                "reference_batch_size = 2",
                "sweep_batch_sizes = [2]",
                "history_plies = 4",
                "batches = 1",
                "warmup_batches = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _run_worker_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the replicate subprocess without paying for an interpreter.

    The command is still the real one and the worker is still the real handler;
    only the process boundary is stubbed, because a CPU suite cannot afford one
    Torch import per replicate.
    """

    import contextlib
    import io
    import subprocess

    import anthro_chess.evaluation.execution_noise as execution_noise_module

    real_run = subprocess.run

    def fake_run(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if command[1:3] != ["-m", "anthro_chess"]:
            # Provenance capture shells out to git; only the replicate worker
            # is being stood in for here.
            return cast(
                "subprocess.CompletedProcess[str]",
                real_run(command, **kwargs),
            )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(command[3:])
        return subprocess.CompletedProcess(
            command,
            returncode=code,
            stdout=stdout.getvalue(),
            stderr="",
        )

    monkeypatch.setattr(execution_noise_module.subprocess, "run", fake_run)


def test_eval_noise_sample_measures_repeatedly_and_records_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=21)
    config = _inference_config(tmp_path / "inference.toml", checkpoint)
    monkeypatch_store = tmp_path / "results"

    assert (
        main(["eval", "noise", "sample", "--config", str(config), "--repeats", "2"])
        == 0
    )

    output = capsys.readouterr().out
    assert "2 reading(s) in this process, recorded nowhere" in output
    assert "inference.move_latency_p50_ms" in output
    assert not monkeypatch_store.exists()


def test_eval_noise_characterizes_this_machines_timing_noise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=22)
    config = _inference_config(tmp_path / "inference.toml", checkpoint)
    store = str(tmp_path / "results")
    _run_worker_in_process(monkeypatch)

    assert (
        main(
            [
                "eval",
                "noise",
                "characterize",
                "--store",
                store,
                "--kind",
                "execution",
                "--config",
                str(config),
                "--processes",
                "2",
                "--repeats",
                "1",
                "--source",
                "two processes on the test machine",
            ]
        )
        == 0
    )
    characterized = capsys.readouterr().out
    assert "Characterized execution noise over 2 reading(s) in 2 process(es)" in (
        characterized
    )
    assert "Valid on:" in characterized

    assert main(["eval", "noise", "list", "--store", store]) == 0
    listed = capsys.readouterr().out
    assert "execution" in listed
    assert "valid on" in listed
    assert "inference.move_latency_p50_ms" in listed


def test_eval_noise_characterize_execution_needs_a_workload_to_repeat(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "eval",
                "noise",
                "characterize",
                "--store",
                str(tmp_path / "results"),
                "--kind",
                "execution",
                "--source",
                "this machine",
            ]
        )
        == 2
    )
    assert "needs --config" in capsys.readouterr().err


def test_eval_noise_characterize_execution_takes_no_checkpoint_list(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An execution floor describes a machine, not a set of checkpoints."""

    assert (
        main(
            [
                "eval",
                "noise",
                "characterize",
                "--store",
                str(tmp_path / "results"),
                "--kind",
                "execution",
                "--config",
                str(tmp_path / "inference.toml"),
                "--checkpoint",
                "checkpoint-a",
                "--source",
                "this machine",
            ]
        )
        == 2
    )
    assert "takes no --checkpoint" in capsys.readouterr().err


def test_eval_noise_characterize_reads_other_kinds_from_the_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only execution noise is measured; the rest are already recorded."""

    assert (
        main(
            [
                "eval",
                "noise",
                "characterize",
                "--store",
                str(tmp_path / "results"),
                "--kind",
                "training",
                "--config",
                str(tmp_path / "inference.toml"),
                "--source",
                "three seeds",
            ]
        )
        == 2
    )
    assert "reads replicates the store already holds" in capsys.readouterr().err
