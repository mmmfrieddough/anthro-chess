import json
import os
import re
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

    (shard,) = (output / "normalized").glob("games-*.parquet")
    assert pq.read_table(shard).num_rows == 1
    command_output = capsys.readouterr().out
    assert "Prepared 1 game(s); rejected 0." in command_output
    assert "Corpus: 1 game(s) from 1 archive(s)." in command_output
    assert "manifests/manifest.json" in command_output


def test_data_prepare_decodes_on_the_workers_it_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag reaches preparation, and its absence leaves the reader a core."""

    import anthro_chess.data as data
    from anthro_chess.interfaces.cli import _prepare_workers

    repository_root = Path(__file__).parents[2]
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"
    config = repository_root / "configs/data/lichess-sample.toml"
    requested: list[int] = []

    def capture(
        _input_path: Path,
        output: Path,
        _resolved: object,
        *,
        workers: int,
        counts_path: Path | None,
    ) -> PreparationResult:
        requested.append(workers)
        return PreparationResult(
            normalized_paths=(output / "normalized/games-0.parquet",),
            manifest_path=output / "manifests/manifest.json",
            accepted_games=1,
            rejected_games=0,
            split_counts={"train": 1, "validation": 0},
            corpus_archives=1,
        )

    monkeypatch.setattr(data, "prepare_pgn", capture)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False
    )
    argv = ["data", "prepare", str(sample), str(tmp_path), "--config", str(config)]

    assert main([*argv, "--workers", "3"]) == 0
    assert main(argv) == 0
    with pytest.raises(SystemExit):
        main([*argv, "--workers", "-4"])

    assert requested == [3, 7]
    assert _prepare_workers(0) == 0


def test_the_decoding_pool_stops_growing_where_one_reader_stops_feeding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bigger machine buys a bigger pool only up to what the reader frames.

    The reader is a single process, so past the point where the pool consumes
    games as fast as one core frames them, another decoder waits rather than
    works.
    """

    from anthro_chess.interfaces.cli import _MAXIMUM_PREPARE_WORKERS, _prepare_workers

    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False
    )

    assert _prepare_workers(None) == _MAXIMUM_PREPARE_WORKERS
    assert _prepare_workers(31) == 31


def test_archives_prepared_at_once_divide_the_machine_rather_than_each_take_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the default forks a full pool per archive onto one machine."""

    from anthro_chess.interfaces.cli import _MAXIMUM_PREPARE_WORKERS, _prepare_workers

    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(32)), raising=False
    )

    assert _prepare_workers(None, 1) == _MAXIMUM_PREPARE_WORKERS
    assert _prepare_workers(None, 8) == 3
    assert _prepare_workers(None, 64) == 0
    assert _prepare_workers(5, 8) == 5


@pytest.mark.parametrize(
    ("cores", "archives", "workers"),
    [(8, 1, 7), (32, 4, 7), (64, 8, 7)],
)
def test_the_default_prepares_the_fewest_archives_that_fill_the_machine(
    monkeypatch: pytest.MonkeyPatch,
    cores: int,
    archives: int,
    workers: int,
) -> None:
    """One archive cannot fill a machine, and its reader caps the pool that can.

    Whatever is chosen has to come to the machine's own process count, since
    both fewer and more were measured slower than that.
    """

    from anthro_chess.interfaces.cli import _prepare_concurrency, _prepare_workers

    monkeypatch.setattr(
        os, "sched_getaffinity", lambda _pid: set(range(cores)), raising=False
    )

    assert _prepare_concurrency(None) == archives
    assert _prepare_workers(None, archives) == workers
    assert archives * (workers + 1) == cores
    assert _prepare_concurrency(3) == 3


def test_data_prepare_reports_an_archive_the_corpus_already_holds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Re-running the command over a prepared archive says so and adds nothing."""

    repository_root = Path(__file__).parents[2]
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"
    config = repository_root / "configs/data/lichess-sample.toml"
    output = tmp_path / "artifacts"
    command = ["data", "prepare", str(sample), str(output), "--config", str(config)]
    assert main(command) == 0
    capsys.readouterr()

    assert main(command) == 0

    command_output = capsys.readouterr().out
    assert "Archive already in this corpus, contributing 1 game(s)." in command_output
    assert "Corpus: 1 game(s) from 1 archive(s)." in command_output


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
    (shard,) = (output / "normalized").glob("games-*.parquet")
    assert pq.read_table(shard).num_rows == 1
    assert "Prepared 1 game(s); rejected 0." in capsys.readouterr().out


def test_data_acquire_command_routes_to_importable_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-blitz-2017-04.toml"
    archive_path = tmp_path / "raw/archive.pgn.zst"
    monkeypatch.setattr(
        "anthro_chess.data.acquire_configured_archive",
        lambda output, archive: AcquisitionResult(
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


def test_data_acquire_fetches_every_archive_into_its_own_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each archive resolves its own directory, so selections can share files."""

    config = tmp_path / "two.toml"
    config.write_text(
        f"""
artifact_name = "fixture"

[source]
id = "test"
version = "fixture"
url = "https://example.test/"
license = "CC0-1.0"

[[archives]]
artifact_name = "month-one"
url = "https://example.test/one.pgn.zst"
file_name = "one.pgn.zst"
sha256 = "{"1" * 64}"

[[archives]]
artifact_name = "month-two"
url = "https://example.test/two.pgn.zst"
file_name = "two.pgn.zst"
sha256 = "{"2" * 64}"
""".lstrip(),
        encoding="utf-8",
    )
    data_root = tmp_path / "datasets"
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))
    seen: list[tuple[Path, str]] = []

    def fake_acquire(output: Path, archive: object) -> AcquisitionResult:
        file_name = archive.file_name  # type: ignore[attr-defined]
        seen.append((output, file_name))
        return AcquisitionResult(
            archive_path=output / "raw" / file_name,
            sha256="a" * 64,
            size_bytes=1,
            reused=False,
        )

    monkeypatch.setattr("anthro_chess.data.acquire_configured_archive", fake_acquire)

    assert main(["data", "acquire", "--config", str(config)]) == 0

    assert seen == [
        (data_root / "month-one", "one.pgn.zst"),
        (data_root / "month-two", "two.pgn.zst"),
    ]
    assert "2 archive(s) verified" in capsys.readouterr().out


def test_data_acquire_uses_data_root_when_output_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-blitz-2017-04.toml"
    data_root = tmp_path / "datasets"
    captured_output: list[Path] = []
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))

    def fake_acquire(output: Path, _archive: object) -> AcquisitionResult:
        captured_output.append(output)
        return AcquisitionResult(
            archive_path=output / "raw/archive.pgn.zst",
            sha256="a" * 64,
            size_bytes=123,
            reused=False,
        )

    monkeypatch.setattr("anthro_chess.data.acquire_configured_archive", fake_acquire)

    assert main(["data", "acquire", "--config", str(config)]) == 0
    assert captured_output == [data_root / "lichess-blitz-2017-04"]


def _census_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Acquire two small archives under a data root and return the selection."""

    from hashlib import sha256

    import zstandard

    def _game(white: str, black: str) -> str:
        return (
            f'[Event "Rated Blitz game"]\n[White "{white}"]\n[Black "{black}"]\n\n'
            "1. e4 e5 1-0\n"
        )

    data_root = tmp_path / "datasets"
    digests = []
    for index, games in enumerate(
        (
            [("Busy", "Quiet"), ("Busy", "Middling")],
            [("Busy", "Middling")],
        ),
        start=1,
    ):
        archive = data_root / f"month-{index}" / "raw" / f"{index}.pgn.zst"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(
            zstandard.ZstdCompressor().compress(
                "".join(_game(*pair) for pair in games).encode()
            )
        )
        digests.append(sha256(archive.read_bytes()).hexdigest())

    config = tmp_path / "selection.toml"
    config.write_text(
        f"""
artifact_name = "fixture"

[source]
id = "test"
version = "fixture"
url = "https://example.test/"
license = "CC0-1.0"

[[archives]]
artifact_name = "month-1"
url = "https://example.test/1.pgn.zst"
file_name = "1.pgn.zst"
sha256 = "{digests[0]}"

[[archives]]
artifact_name = "month-2"
url = "https://example.test/2.pgn.zst"
file_name = "2.pgn.zst"
sha256 = "{digests[1]}"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    return config


def test_data_census_keeps_asking_about_an_archive_that_was_reclaimed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counts outlive the archive, so deleting a prepared one costs nothing."""

    config = _census_fixture(tmp_path, monkeypatch)
    asked: list[str] = []

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        asked.extend(batch)
        return [{"id": name} for name in batch]

    monkeypatch.setattr("anthro_chess.data.census._post_usernames", fake_post)
    command = [
        "data",
        "census",
        "--config",
        str(config),
        "--pause-seconds",
        "0",
        "--workers",
        "1",
        "--accounts",
    ]
    assert main([*command, "1"]) == 0
    capsys.readouterr()

    (tmp_path / "datasets/month-1/raw/1.pgn.zst").unlink()
    assert main([*command, "2"]) == 0

    # The deleted archive still contributes its accounts and its counts, so the
    # queue and the coverage denominators are what they were.
    assert asked == ["busy", "middling", "quiet"]
    assert "Coverage: 3 of 3 account(s) (100.00%), 100.00% of player-slots" in (
        capsys.readouterr().out
    )


def test_data_census_spends_the_allowance_before_it_counts_a_new_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A day's allowance does not roll over; hours of counting can wait."""

    config = _census_fixture(tmp_path, monkeypatch)
    asked: list[str] = []

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        asked.extend(batch)
        return [{"id": name} for name in batch]

    monkeypatch.setattr("anthro_chess.data.census._post_usernames", fake_post)
    command = [
        "data",
        "census",
        "--config",
        str(config),
        "--pause-seconds",
        "0",
        "--workers",
        "1",
    ]
    counts = tmp_path / "datasets/month-2/census/2.pgn.zst.accounts.tsv"

    # Counting comes first only where there is nothing to ask about yet.
    assert main([*command, "--accounts", "1"]) == 0
    assert asked == ["busy"]
    capsys.readouterr()

    # A newly acquired archive waits behind the day's asking rather than in
    # front of it, because the allowance is what does not roll over.
    counts.unlink()
    assert main([*command, "--accounts", "1"]) == 0

    assert asked == ["busy", "middling"]
    assert counts.is_file()


def test_data_census_asks_the_busiest_accounts_first_and_stores_every_answer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _census_fixture(tmp_path, monkeypatch)
    asked: list[str] = []

    def fake_post(batch: list[str], token: str | None) -> list[dict[str, object]]:
        asked.extend(batch)
        return [{"id": name, "tosViolation": name == "busy"} for name in batch]

    monkeypatch.setattr("anthro_chess.data.census._post_usernames", fake_post)

    assert (
        main(
            [
                "data",
                "census",
                "--config",
                str(config),
                "--accounts",
                "2",
                "--pause-seconds",
                "0",
                "--workers",
                "1",
            ]
        )
        == 0
    )

    # Busiest first across both archives, and the counts stay beside each one.
    assert asked == ["busy", "middling"]
    data_root = tmp_path / "datasets"
    assert (data_root / "month-1/census/1.pgn.zst.accounts.tsv").is_file()
    assert (data_root / "month-2/census/2.pgn.zst.accounts.tsv").is_file()
    # The answers are the source's, not the selection's, so another selection
    # over the same source inherits them.
    answers = (data_root / "test-account-census/answers.tsv").read_text(
        encoding="utf-8"
    )
    assert answers.splitlines()[0].startswith("busy\t1\t")
    output = capsys.readouterr().out
    assert (
        "Asked about 2 account(s); 1 marked. The requested accounts are asked about."
        in output
    )
    assert "Coverage: 2 of 3 account(s) (66.67%), 83.33% of player-slots" in output


def test_data_mark_accounts_cuts_a_snapshot_from_the_census_as_it_stands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthro_chess.data.accounts import load_marked_accounts

    config = _census_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "anthro_chess.data.census._post_usernames",
        lambda batch, token: [
            {"id": name, "tosViolation": name == "busy"} for name in batch
        ],
    )
    output_path = tmp_path / "marked.txt"
    command = ["data", "mark-accounts", "--config", str(config), "--output"]

    # A snapshot the census cannot speak for would stop preparation partway
    # through a corpus that cannot be repaired one archive at a time.
    assert main([*command, str(output_path)]) == 2
    assert "counted 0 of this selection's 2 archive(s)" in capsys.readouterr().err

    assert (
        main(
            [
                "data",
                "census",
                "--config",
                str(config),
                "--pause-seconds",
                "0",
                "--workers",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main([*command, str(output_path)]) == 0

    snapshot = load_marked_accounts(output_path)
    assert snapshot.contains("BUSY")
    assert snapshot.accounts_queried == 3
    assert snapshot.slot_coverage == 1.0
    assert len(snapshot.covers_archives) == 2
    assert "Coverage: 100.00% of accounts" in capsys.readouterr().out

    # A snapshot cut from a later census rejects games this one keeps, so it is
    # a new corpus rather than an overwrite.
    assert main([*command, str(output_path)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_data_prepare_infers_shared_archive_independently_of_prepared_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-blitz-2017-04.toml"
    data_root = tmp_path / "datasets"
    captured_paths: list[tuple[Path, Path, Path | None]] = []
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(data_root))

    def fake_prepare(
        input_path: Path,
        output: Path,
        _resolved: object,
        *,
        workers: int,
        counts_path: Path | None,
    ) -> PreparationResult:
        captured_paths.append((input_path, output, counts_path))
        return PreparationResult(
            normalized_paths=(output / "normalized/games-0.parquet",),
            manifest_path=output / "manifests/manifest.json",
            accepted_games=1,
            rejected_games=0,
            split_counts={"train": 1, "validation": 0},
            corpus_archives=1,
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
    # The corpus is named for this run; the archive and the account counts it
    # leaves behind belong to the archive, which selections share.
    assert captured_paths == [
        (
            data_root / "lichess-blitz-2017-04/raw/"
            "lichess_db_standard_rated_2017-04.pgn.zst",
            data_root / "proof-slice",
            data_root / "lichess-blitz-2017-04/census/"
            "lichess_db_standard_rated_2017-04.pgn.zst.accounts.tsv",
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
        output_directory: Path,
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


@pytest.mark.parametrize(
    ("extra_argv", "expected_name"),
    [
        ([], "example-run"),
        # The shape that used to escape the run root, pinned rather than left
        # resting on the absence of an opt-out branch.
        (["--set", 'run_name="probe"'], "probe"),
    ],
    ids=["declared-in-config", "named-on-the-command-line"],
)
def test_train_uses_machine_roots_for_checked_in_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    expected_name: str,
) -> None:
    config = tmp_path / "training.toml"
    config.write_text(
        """
run_name = "example-run"

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
        output_directory: Path,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        assert output_directory == run_root / expected_name
        assert resolved.value.train.normalized == data_root / "example-data/normalized"
        assert (
            resolved.value.train.manifest
            == data_root / "example-data/manifests/manifest.json"
        )
        return TrainingResult(
            run_path=run_root / expected_name / "run.json",
            metrics_path=run_root / expected_name / "metrics.jsonl",
            checkpoint_path=run_root / expected_name / "checkpoints/step-00000001.pt",
            steps=1,
            initial_parameter_sha256="a",
            final_parameter_sha256="b",
            validation=None,
        )

    monkeypatch.setattr("anthro_chess.training.run_training", fake_run)

    assert main(["train", "--config", str(config), *extra_argv]) == 0


def test_train_roots_a_shard_backed_selection_and_keeps_how_it_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "training.toml"
    config.write_text(
        """
run_name = "example-run"

[train]
normalized = "artifacts/example-data/normalized"
manifest = "artifacts/example-data/manifests/manifest.json"

[train.loader]
split = "train"

[train.streaming]
planning_window_examples = 32
workers = 2
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
        output_directory: Path,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        selection = resolved.value.train
        assert selection.normalized == data_root / "example-data/normalized"
        # Rooting rewrites a selection's paths and has to carry the rest of it
        # across, or a rooted run would quietly fall back to eager loading.
        assert selection.streaming is not None
        assert selection.streaming.planning_window_examples == 32
        assert selection.streaming.workers == 2
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
        output_directory: Path,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        assert output_directory == explicit_output
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
                "--output-directory",
                str(explicit_output),
                "--set",
                f'train.normalized="{explicit_normalized}"',
                "--set",
                f'train.manifest="{explicit_manifest}"',
            ]
        )
        == 0
    )


def test_train_without_a_run_root_writes_beneath_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh clone still resolves a named run inside the checkout."""

    config = tmp_path / "training.toml"
    config.write_text(
        """
run_name = "example-run"

[train]
normalized = "artifacts/example-data/normalized"
manifest = "artifacts/example-data/manifests/manifest.json"

[train.loader]
split = "train"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHRO_CHESS_RUN_ROOT", raising=False)
    monkeypatch.delenv("ANTHRO_CHESS_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    def fake_run(
        resolved: ResolvedConfig[TrainingConfig],
        *,
        output_directory: Path,
        store: object = None,
        detail: object = None,
    ) -> TrainingResult:
        assert output_directory == Path("artifacts/example-run")
        return TrainingResult(
            run_path=Path("artifacts/example-run/run.json"),
            metrics_path=Path("artifacts/example-run/metrics.jsonl"),
            checkpoint_path=Path("artifacts/example-run/checkpoints/step-1.pt"),
            steps=1,
            initial_parameter_sha256="a",
            final_parameter_sha256="b",
            validation=None,
        )

    monkeypatch.setattr("anthro_chess.training.run_training", fake_run)

    assert main(["train", "--config", str(config)]) == 0


@pytest.mark.parametrize("run_name", ["../escape", "nested/run", "..", "/absolute"])
def test_train_refuses_a_run_name_that_is_not_one_path_component(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    run_name: str,
) -> None:
    """A name places a run inside the root; it does not get to leave it."""

    config = tmp_path / "training.toml"
    config.write_text(
        f"""
run_name = "{run_name}"

[train]
normalized = "artifacts/example-data/normalized"
manifest = "artifacts/example-data/manifests/manifest.json"

[train.loader]
split = "train"
""",
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config)]) == 2
    assert "run_name" in capsys.readouterr().err


#: Shared by both fixture readings, so they compare as two checkpoints of one
#: configuration.
CLI_TRAINING_SHA256 = "7d" * 32


def _record_fixture_results(
    store_root: Path,
    *,
    training_sha256: str | None = CLI_TRAINING_SHA256,
    month: int = 7,
) -> None:
    """Record two comparable results so the report command has history.

    ``month`` moves the whole pair earlier, so a caller can lay down an older
    generation of the same two readings under the store's append-only history.
    """

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
                checkpoint=CheckpointReference(
                    label=label,
                    training_sha256=training_sha256,
                ),
                data=dataset_reference(
                    pool_id="fixture-pool",
                    pool_version=1,
                    view="canonical",
                    selected_games=component.games,
                    game_ids_sha256="a" * 64,
                    components=[component],
                ),
                measurements=[measurement("held_out.move_loss", value, data=component)],
                recorded_at=datetime(2026, month, day, tzinfo=UTC),
            )
        )


#: Every `anthro eval` subcommand that runs one benchmark from a selection file.
_BENCHMARK_SUBCOMMANDS = (
    "run",
    "puzzles",
    "novelty",
    "inference",
    "rollout",
    "termination",
    "ladder",
)


@pytest.mark.parametrize("subcommand", _BENCHMARK_SUBCOMMANDS)
def test_every_benchmark_subcommand_runs_its_registry_entry(
    tmp_path: Path,
    subcommand: str,
) -> None:
    """Each command names its benchmark, and everything else follows the name.

    A subcommand wired to a name the registry does not hold, or one that goes
    around the registry to call a benchmark directly, would take its schema,
    its rooting, its error types and its view from somewhere else again.
    """

    from anthro_chess.evaluation.benchmarks import benchmark_registry
    from anthro_chess.interfaces.cli import (
        _STEP_RENDERERS,
        _run_eval_benchmark,
        build_parser,
    )

    arguments = build_parser().parse_args(
        ["eval", subcommand, "--config", str(tmp_path / "benchmark.toml")]
    )

    assert arguments.handler.func is _run_eval_benchmark
    assert arguments.handler.keywords["name"] == subcommand
    assert subcommand in benchmark_registry()
    assert subcommand in _STEP_RENDERERS


@pytest.mark.parametrize("subcommand", _BENCHMARK_SUBCOMMANDS)
def test_eval_handlers_share_one_recording_decision(
    tmp_path: Path,
    subcommand: str,
) -> None:
    """A subcommand that drops one of the three recording flags fails here."""

    from anthro_chess.interfaces.cli import _result_stores, build_parser

    parser = build_parser()
    # Parsing is the whole exercise here, so the named config is never read.
    invocation = ["eval", subcommand, "--config", str(tmp_path / "benchmark.toml")]

    withheld = parser.parse_args([*invocation, "--no-record"])
    assert _result_stores(withheld) == (None, None)

    recording = parser.parse_args(
        [
            *invocation,
            "--store",
            str(tmp_path / "results"),
            "--detail-root",
            str(tmp_path / "detail"),
        ]
    )
    store, detail = _result_stores(recording)
    assert store is not None and store.root == tmp_path / "results"
    assert detail is not None and detail.root == tmp_path / "detail"


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


def test_eval_metrics_states_why_a_metric_can_carry_no_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The listing is where a report's "unqualifiable" verdict is explained.

    Without it the verdict is a word with nowhere to look up, which relocates
    the ambiguity it exists to remove rather than removing it.
    """

    assert main(["eval", "metrics"]) == 0

    rendered = " ".join(capsys.readouterr().out.split())
    assert "no sampling floor can exist: the rate counts rating slices" in rendered


def test_eval_metrics_aligns_every_family_on_one_column(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The listing is read top to bottom, so its column spans the families.

    Sizing per family would let a family of short names print a narrower table
    than the one above it, which reads as a different table rather than as the
    rest of the same one.
    """

    assert main(["eval", "metrics"]) == 0

    rendered = capsys.readouterr().out
    # Where the direction column starts on each metric row. It is the first
    # thing right of the identifier, so a row that outgrew the column shows up
    # here first. Matching on a dotted first token skips the family headings
    # and the line a family with no metric yet prints.
    directions = {
        match.end()
        for line in rendered.splitlines()
        if (match := re.match(r"  \S+\.\S+ +", line)) is not None
    }

    assert len(directions) == 1


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


def test_eval_puzzles_reports_the_resolution_it_bought(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    write_puzzle_artifact: Callable[..., Path],
    inference_run: Callable[..., Path],
) -> None:
    """A solve rate beside no resolution has been read as a model finding once.

    So the realized sample size and what it can distinguish are part of the
    command's output rather than something a reader recovers from the config.
    """

    artifact = write_puzzle_artifact(
        tmp_path / "puzzles",
        ratings=(1200, 1400),
        puzzles_per_rating=4,
    )
    normalized, _ = write_corpus(
        tmp_path / "corpus",
        [{**normalized_row(1, split="train"), "source_id": "lichess"}],
    )
    checkpoint = inference_run(tmp_path / "run")
    config = tmp_path / "puzzles.toml"
    config.write_text(
        f'puzzle_set = "{artifact}"\n'
        f'training_normalized = "{normalized}"\n'
        "target_ratings = [1000, 1800]\n"
        "puzzles_per_rating = 2\n"
        "\n[model]\n"
        f'checkpoint_path = "{checkpoint}"\n'
        'device = "cpu"\n',
        encoding="utf-8",
    )

    assert main(["eval", "puzzles", "--config", str(config), "--no-record"]) == 0

    printed = capsys.readouterr().out
    assert "4 of 8 puzzle(s), 2 per rating" in printed
    assert re.search(r"Resolution: \d+\.\d\d pp for independent readings", printed)
    # Decision 0040: the slope and the ordering are what a below-resolution
    # reading was once written up from, so neither is printed unqualified.
    assert re.search(r"Greedy slope=-?\d+\.\d{4} \((±|spread unknown)", printed)
    assert re.search(r"order=\d\.\d{3} \((±|spread unknown)", printed)
    assert "Response resolution: 1000 stratified refits of 4 redrawn puzzle(s)" in (
        printed
    )
    assert re.search(r"fitted puzzle rating: (±\d|spread unknown)", printed)
    assert max(len(text) for text in printed.splitlines()) <= 120


def test_eval_rollout_reports_a_configuration_error_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The prefix arm without a pool is a configuration error, not a skip."""

    config = tmp_path / "rollout.toml"
    config.write_text('arms = ["human-prefix"]\n', encoding="utf-8")

    assert main(["eval", "rollout", "--config", str(config), "--no-record"]) == 2

    assert "anthro eval rollout:" in capsys.readouterr().err


def test_eval_rollout_renders_every_cell_with_its_series(tmp_path: Path) -> None:
    """The text view has to name each cell's series and what it played."""

    import torch

    from anthro_chess.chess import ACTION_VOCABULARY_SIZE
    from anthro_chess.data import DecisionContext
    from anthro_chess.evaluation import RolloutBenchmarkConfig
    from anthro_chess.evaluation.benchmarks import (
        benchmark_registry,
        run_benchmark,
    )
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
    result = run_benchmark(
        benchmark_registry()["rollout"],
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


def test_eval_ladder_renders_the_transfer_function_and_its_error_profile(
    tmp_path: Path,
) -> None:
    """The text view has to show ordering, shape, and the profile beside it."""

    import torch

    from anthro_chess.chess import ACTION_VOCABULARY_SIZE, RESIGNATION_ACTION_ID
    from anthro_chess.data import DecisionContext
    from anthro_chess.evaluation import LadderBenchmarkConfig
    from anthro_chess.evaluation.benchmarks import (
        benchmark_registry,
        run_benchmark,
    )
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
    result = run_benchmark(
        benchmark_registry()["ladder"],
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


def _record_sampled_reading(
    store_root: Path,
    *,
    floor: float,
    games: int,
    selected_games: int | None = None,
    envelope_version: int | None = None,
) -> None:
    """Record the reading an evaluation run leaves behind, spread included.

    ``games`` is what realized the metric and ``selected_games`` what the pass
    scored. They differ only for a sliced metric. ``envelope_version`` stamps
    the record at an older schema version, which is what a store written before
    units became per-metric holds.
    """

    import math
    from datetime import UTC, datetime

    from anthro_chess.evaluation.results import (
        DEFAULT_COVERAGE,
        BenchmarkReference,
        CheckpointReference,
        MetricDispersion,
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
    # Chosen so a delta against a reading like this one faces exactly ``floor``.
    bound = floor / (DEFAULT_COVERAGE * math.sqrt(2.0))
    envelope = build_result(
        kind="held-out-prediction",
        benchmark=BenchmarkReference(name="move-validation", version=1),
        checkpoint=CheckpointReference(label="checkpoint-a"),
        data=dataset_reference(
            pool_id="fixture-pool",
            pool_version=1,
            view="canonical",
            selected_games=games if selected_games is None else selected_games,
            game_ids_sha256="a" * 64,
            components=[component],
        ),
        measurements=[
            measurement(
                "held_out.move_loss",
                3.5,
                data=component,
                sample_size=games,
                dispersion=MetricDispersion(
                    value=bound,
                    bound=bound,
                    units=games,
                    source="the fixture pool",
                ),
            )
        ],
        recorded_at=datetime(2026, 7, 9, tzinfo=UTC),
    )
    if envelope_version is not None:
        envelope = envelope.model_copy(update={"envelope_version": envelope_version})
    ResultsStore(store_root).append(envelope)


def test_eval_noise_plan_sizes_an_axis_from_a_measured_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record_sampled_reading(tmp_path / "results", floor=0.04, games=1_000)

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
    assert plan["required_realizing_games"] == 16_000
    assert plan["measured_realizing_games"] == 1_000
    # Every game realized this metric, so the two counts are one number.
    assert plan["required_pool_games"] == 16_000


def test_eval_noise_plan_scales_a_sliced_metric_up_to_a_pool_size(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The spread was read over the games that realized the slice, so the count
    # extrapolated from it is in those too. A pool has to be larger by the rate
    # it realizes them at, which is the difference between a pool that resolves
    # the effect and one a tenth of the size that does not.
    _record_sampled_reading(
        tmp_path / "results", floor=0.04, games=100, selected_games=1_000
    )

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
    assert plan["required_realizing_games"] == 1_600
    assert plan["required_pool_games"] == 16_000


def test_eval_noise_plan_reports_a_missing_spread_without_a_traceback(
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
    assert "records a sampled dispersion" in capsys.readouterr().err


def test_eval_noise_plan_refuses_a_reading_that_counted_the_whole_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Below envelope version 8 a sliced metric's `units` counted every game the
    # pass scored, so extrapolating from one would size a pool in a unit the
    # record does not carry.
    _record_sampled_reading(
        tmp_path / "results", floor=0.04, games=1_000, envelope_version=7
    )

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
    assert "envelope version 8 or above" in capsys.readouterr().err


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


def test_eval_noise_sample_measures_one_reading_and_records_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    inference_run: Callable[..., Path],
) -> None:
    checkpoint = inference_run(tmp_path / "run", seed=21)
    config = _inference_config(tmp_path / "inference.toml", checkpoint)
    monkeypatch_store = tmp_path / "results"

    assert main(["eval", "noise", "sample", "--config", str(config)]) == 0

    output = capsys.readouterr().out
    assert "One reading in this process, recorded nowhere" in output
    assert "inference.move_latency_p50_ms" in output
    assert not monkeypatch_store.exists()


def test_eval_suite_plans_the_shipped_selection_without_running_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The shipped sweep has to resolve, in order, against the real files."""

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    assert (
        main(
            [
                "eval",
                "suite",
                "--config",
                "configs/evaluation/checkpoint-suite.toml",
                "--plan",
                "--format",
                "json",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    names = [step["benchmark"] for step in plan["steps"]]
    assert plan["scale"] == "reduced"
    # Decision decomposition reads the games the rollout played, so it can
    # never be planned ahead of it.
    assert names.index("decisions") > names.index("rollout")
    assert set(names) == {
        "inference",
        "run",
        "novelty",
        "puzzles",
        "rollout",
        "decisions",
        "termination",
        "ladder",
    }
    decisions = next(step for step in plan["steps"] if step["benchmark"] == "decisions")
    assert decisions["record"] is False
    assert decisions["reads_games_from"] == "rollout"


def test_eval_suite_full_scale_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sweep measured in hours is not a default anyone would run."""

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    arguments = [
        "eval",
        "suite",
        "--config",
        "configs/evaluation/checkpoint-suite.toml",
        "--plan",
        "--format",
        "json",
    ]

    assert main(arguments) == 0
    reduced = json.loads(capsys.readouterr().out)
    assert main([*arguments, "--full"]) == 0
    full = json.loads(capsys.readouterr().out)

    assert (reduced["scale"], full["scale"]) == ("reduced", "full")
    reduced_run = next(s for s in reduced["steps"] if s["benchmark"] == "run")
    full_run = next(s for s in full["steps"] if s["benchmark"] == "run")
    assert "view.maximum_games=400" in reduced_run["overrides"]
    assert full_run["overrides"] == []
    # The two scales differ by what each step reads, not by which steps run.
    reduced_names = {step["benchmark"] for step in reduced["steps"]}
    full_names = {step["benchmark"] for step in full["steps"]}
    assert reduced_names == full_names
    # Two scales are two series, so a resume can never cross between them.
    assert reduced["plan_sha256"] != full["plan_sha256"]


def test_eval_suite_threads_one_checkpoint_through_every_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    assert (
        main(
            [
                "eval",
                "suite",
                "--config",
                "configs/evaluation/checkpoint-suite.toml",
                "--set",
                'model.run_path="training-blitz-30k-v4"',
                "--set",
                'model.checkpoint="step-00008000.pt"',
                "--plan",
                "--format",
                "json",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    configured = [step for step in plan["steps"] if step["benchmark"] != "decisions"]
    assert configured
    for step in configured:
        assert 'model.run_path="training-blitz-30k-v4"' in step["overrides"]
        assert 'model.checkpoint="step-00008000.pt"' in step["overrides"]


def test_eval_suite_reports_a_configuration_error_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "suite.toml"
    config.write_text(
        '[benchmarks.nonsense]\nconfig = "missing.toml"\n', encoding="utf-8"
    )

    assert main(["eval", "suite", "--config", str(config), "--plan"]) == 2

    assert "anthro eval suite: unknown benchmark" in capsys.readouterr().err


def test_eval_suite_needs_somewhere_to_keep_its_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Resume depends on the ledger, so a missing home fails loudly."""

    monkeypatch.delenv("ANTHRO_CHESS_RUN_ROOT", raising=False)
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    config = tmp_path / "suite.toml"
    config.write_text(
        "[benchmarks.inference]\n"
        'config = "configs/evaluation/inference-efficiency.toml"\n',
        encoding="utf-8",
    )

    assert main(["eval", "suite", "--config", str(config), "--no-record"]) == 2

    assert "--sweep-root" in capsys.readouterr().err


def test_eval_suite_shrinks_the_ladder_rather_than_dropping_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reduction is reached at one scale and absent at the other.

    Against the shipped selection rather than a fixture, because the state this
    pins against is a `reduced` list beside a `scales` that drops the step: the
    overrides then reach no schema at either scale and nothing finds them wrong.
    """

    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    arguments = [
        "eval",
        "suite",
        "--config",
        "configs/evaluation/checkpoint-suite.toml",
        "--plan",
        "--format",
        "json",
    ]

    assert main(arguments) == 0
    reduced = json.loads(capsys.readouterr().out)
    assert main([*arguments, "--full"]) == 0
    full = json.loads(capsys.readouterr().out)

    reduced_ladder = next(s for s in reduced["steps"] if s["benchmark"] == "ladder")
    full_ladder = next(s for s in full["steps"] if s["benchmark"] == "ladder")
    assert "grid.seeds=[0]" in reduced_ladder["overrides"]
    assert "openings.view.maximum_games=4" in reduced_ladder["overrides"]
    assert full_ladder["overrides"] == []


def _retained_run(
    root: Path,
    name: str,
    record: dict[str, Any],
    step: int = 8000,
) -> Path:
    """Write the marker files a run is recognized and selected by.

    No weights are written, because neither the report nor the selection
    record reads any: the report compares the run record against the contract
    this code loads, and the selection record checks the two files exist.
    """

    checkpoints = root / name / "checkpoints"
    checkpoints.mkdir(parents=True)
    (root / name / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (checkpoints / f"step-{step:08d}.pt").write_bytes(b"")
    (checkpoints / "latest.json").write_text(
        json.dumps({"global_step": step, "path": f"step-{step:08d}.pt"}),
        encoding="utf-8",
    )
    return root / name


def _unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "ANTHRO_CHESS_DATA_ROOT",
        "ANTHRO_CHESS_RUN_ROOT",
        "ANTHRO_CHESS_RESULTS_ROOT",
        "ANTHRO_CHESS_RESULT_DETAIL_ROOT",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_machine_reports_the_runs_and_artifacts_the_roots_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    loadable_run_record: dict[str, Any],
) -> None:
    _unconfigured(monkeypatch)
    _retained_run(tmp_path / "runs", "trained", loadable_run_record)
    (tmp_path / "datasets" / "corpus" / "manifests").mkdir(parents=True)
    (tmp_path / "datasets" / "corpus" / "manifests" / "manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    assert main(["machine"]) == 0

    output = capsys.readouterr().out
    assert "1 checkpoint(s), step-00008000.pt  loadable" in output
    assert "corpus  corpus" in output


def test_machine_says_which_retained_runs_this_code_can_still_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    loadable_run_record: dict[str, Any],
) -> None:
    """Trying runs one at a time is what this replaces, at 14 seconds a miss."""

    _unconfigured(monkeypatch)
    runs = tmp_path / "runs"
    _retained_run(runs, "current", loadable_run_record)
    retired = _retained_run(runs, "retired", loadable_run_record)
    (retired / "run.json").write_text(
        json.dumps(
            {
                **loadable_run_record,
                "model": {**loadable_run_record["model"], "version": 4},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(runs))
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    (tmp_path / "datasets").mkdir()

    assert main(["machine"]) == 0

    lines = {
        line.split()[0]: line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("  current") or line.startswith("  retired")
    }
    assert lines["current"].endswith("  loadable")
    # The reason, so a stale run root is distinguishable from a broken install.
    assert "not loadable: model identity 4" in lines["retired"]


def test_machine_reports_a_half_configured_pair_as_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact state that once cost a session a shakedown reading."""

    _unconfigured(monkeypatch)
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path))

    assert main(["machine"]) == 1

    output = capsys.readouterr().out
    assert "problems" in output
    assert "ANTHRO_CHESS_RUN_ROOT is not" in output


def test_machine_reports_a_fresh_clone_as_configured_nothing_rather_than_broken(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _unconfigured(monkeypatch)

    assert main(["machine"]) == 0

    output = capsys.readouterr().out
    assert "problems" not in output
    assert (
        "not set; configured relative paths resolve in the working directory" in output
    )


def test_machine_json_carries_the_whole_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    loadable_run_record: dict[str, Any],
) -> None:
    _unconfigured(monkeypatch)
    _retained_run(tmp_path / "runs", "trained", loadable_run_record)
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))

    assert main(["machine", "--format", "json"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert [run["name"] for run in report["runs"]] == ["trained"]
    # The data root is set to a directory that was never created, which the
    # report has to distinguish from a data root holding nothing.
    assert len(report["problems"]) == 1


def test_model_select_records_the_machine_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    loadable_run_record: dict[str, Any],
) -> None:
    _unconfigured(monkeypatch)
    _retained_run(tmp_path / "runs", "trained", loadable_run_record)
    monkeypatch.setenv("ANTHRO_CHESS_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHRO_CHESS_DATA_ROOT", str(tmp_path / "datasets"))
    (tmp_path / "datasets").mkdir()

    assert main(["model", "select", "trained"]) == 0
    capsys.readouterr()
    assert main(["machine"]) == 0

    output = capsys.readouterr().out
    assert str(tmp_path / "runs/trained/checkpoints/step-00008000.pt") in output


def test_model_select_without_a_run_root_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _unconfigured(monkeypatch)

    assert main(["model", "select", "trained"]) == 2

    error = capsys.readouterr().err
    assert "--run-root" in error
    assert "ANTHRO_CHESS_RUN_ROOT" in error
    assert "retained training runs" in error


def test_a_missing_data_root_says_what_it_would_have_to_hold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _unconfigured(monkeypatch)
    repository_root = Path(__file__).parents[2]
    config = repository_root / "configs/data/lichess-sample.toml"
    sample = repository_root / "samples/lichess/standard-export-sample.pgn"

    assert main(["data", "prepare", str(sample), "--config", str(config)]) == 2

    error = capsys.readouterr().err
    assert "ANTHRO_CHESS_DATA_ROOT" in error
    assert "corpora" in error


def test_eval_puzzles_says_when_it_estimated_no_response_resolution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
    write_puzzle_artifact: Callable[..., Path],
    inference_run: Callable[..., Path],
) -> None:
    """A run that could not estimate a resolution has to say so, not go quiet."""

    artifact = write_puzzle_artifact(
        tmp_path / "puzzles",
        ratings=(1200, 1400),
        puzzles_per_rating=4,
    )
    normalized, _ = write_corpus(
        tmp_path / "corpus",
        [{**normalized_row(1, split="train"), "source_id": "lichess"}],
    )
    checkpoint = inference_run(tmp_path / "run")
    config = tmp_path / "puzzles.toml"
    config.write_text(
        f'puzzle_set = "{artifact}"\n'
        f'training_normalized = "{normalized}"\n'
        "target_ratings = [1000, 1800]\n"
        "\n[noise]\n"
        "enabled = false\n"
        "\n[model]\n"
        f'checkpoint_path = "{checkpoint}"\n'
        'device = "cpu"\n',
        encoding="utf-8",
    )

    assert main(["eval", "puzzles", "--config", str(config), "--no-record"]) == 0

    printed = capsys.readouterr().out
    assert "Response resolution: not estimated" in printed
    assert "(±" not in printed
