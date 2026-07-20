from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from anthro_chess import __version__
from anthro_chess.config import ResolvedConfig
from anthro_chess.data import AcquisitionResult
from anthro_chess.interfaces.cli import main
from anthro_chess.training import TrainingConfig, TrainingResult


def test_smoke_command_needs_no_external_resources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke"]) == 0
    assert capsys.readouterr().out == (
        f"Anthro Chess {__version__} is installed and ready.\n"
    )


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
    for planned_command in ("evaluate", "play", "uci"):
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

    def fake_run(resolved: ResolvedConfig[TrainingConfig]) -> TrainingResult:
        assert resolved.value.steps == 2
        return TrainingResult(
            run_path=run_path,
            metrics_path=metrics_path,
            steps=2,
            initial_parameter_sha256="a",
            final_parameter_sha256="b",
            validation=None,
        )

    monkeypatch.setattr("anthro_chess.training.run_training", fake_run)

    assert main(["train", "--config", str(config), "--set", "steps=2"]) == 0

    command_output = capsys.readouterr().out
    assert "Completed 2 optimizer step(s)." in command_output
    assert f"Run: {run_path}" in command_output
    assert f"Metrics: {metrics_path}" in command_output
