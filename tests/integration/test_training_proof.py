from __future__ import annotations

import json
from pathlib import Path

from anthro_chess.interfaces.cli import main
from anthro_chess.training import load_training_checkpoint

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_cpu_proof_prepares_trains_validates_checkpoints_and_resumes(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    sample = REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn"
    data_config = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"
    training_config = REPOSITORY_ROOT / "configs/training/sample-smoke.toml"

    assert (
        main(
            [
                "data",
                "prepare",
                str(sample),
                str(data_root),
                "--config",
                str(data_config),
            ]
        )
        == 0
    )

    # Every run records what it cost, so the proof has to name a store or it
    # would append to the repository's committed history on each test run.
    common_overrides = [
        "--store",
        str(tmp_path / "results"),
        "--detail-root",
        str(tmp_path / "detail"),
        "--output-directory",
        str(run_root),
        "--set",
        f"train.normalized={json.dumps(str(data_root / 'normalized'))}",
        "--set",
        f"train.manifest={json.dumps(str(data_root / 'manifests/manifest.json'))}",
        "--set",
        f"validation.normalized={json.dumps(str(data_root / 'normalized'))}",
        "--set",
        (
            "validation.manifest="
            f"{json.dumps(str(data_root / 'manifests/manifest.json'))}"
        ),
    ]
    assert main(["train", "--config", str(training_config), *common_overrides]) == 0
    initial_record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert (
        main(
            [
                "train",
                "--config",
                str(training_config),
                *common_overrides,
                "--set",
                'resume_from="latest"',
                "--set",
                "steps=6",
            ]
        )
        == 0
    )

    resumed_record = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    validation = resumed_record["validation"]
    checkpoint = load_training_checkpoint(
        Path(resumed_record["optimization"]["checkpoint"])
    )
    metric_steps = [
        json.loads(line)["global_step"]
        for line in (run_root / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert initial_record["optimization"]["completed_steps"] == 3
    assert resumed_record["optimization"]["starting_step"] == 3
    assert resumed_record["optimization"]["completed_steps"] == 6
    assert (
        resumed_record["optimization"]["initial_parameter_sha256"]
        == (initial_record["optimization"]["final_parameter_sha256"])
    )
    assert validation["position_count"] > 0
    assert validation["version"] == 2
    assert validation["legal_move_loss"] > 0.0
    assert validation["baselines"]["uniform_over_legal_move_loss"] > 0.0
    assert validation["ratings"]["rated_position_count"] > 0
    assert validation["legality"]["mask_penalty"] >= 0.0
    assert checkpoint["global_step"] == 6
    # A resumed run keeps counting the positions the checkpoint already paid
    # for, so the budget axis a quality-versus-time report reads is the whole
    # run rather than the segment since the restart.
    efficiency = resumed_record["efficiency"]
    assert (
        efficiency["processed_positions"]
        == (resumed_record["optimization"]["processed_positions"])
    )
    assert (
        efficiency["processed_positions"]
        > (initial_record["efficiency"]["processed_positions"])
    )
    assert efficiency["coordinates"]["model_sha256"]
    assert efficiency["coordinates"]["dataset_sha256"]
    assert efficiency["run_seconds"] > efficiency["training_seconds"] > 0.0
    assert efficiency["overhead"]["startup_seconds"] > 0.0
    assert len(efficiency["recorded"]) == 1
    assert len(efficiency["detail"]) == 1
    assert Path(efficiency["detail"][0]).is_file()
    assert checkpoint["metadata"]["action_vocabulary"]["sha256"]
    assert checkpoint["metadata"]["encoding"]["schema_sha256"]
    assert checkpoint["metadata"]["data"]["train"]["manifest_sha256"]
    assert metric_steps == [1, 2, 3, 4, 5, 6]
