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

    def fake_run(resolved: ResolvedConfig[TrainingConfig]) -> TrainingResult:
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

    def fake_run(resolved: ResolvedConfig[TrainingConfig]) -> TrainingResult:
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

    def fake_run(resolved: ResolvedConfig[TrainingConfig]) -> TrainingResult:
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
