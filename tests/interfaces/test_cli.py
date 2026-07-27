import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from anthro_chess import __version__
from anthro_chess.config import ResolvedConfig
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


def test_eval_metrics_lists_the_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["eval", "metrics", "--format", "json"]) == 0

    registry = json.loads(capsys.readouterr().out)
    families = {family["identifier"] for family in registry["families"]}
    assert {"training-health", "legality", "rating-behavior", "generated-play"} <= (
        families
    )


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
