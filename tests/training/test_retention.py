import json
from pathlib import Path

import pytest

from anthro_chess.inference.selection import write_model_selection
from anthro_chess.interfaces.cli import main
from anthro_chess.training.retention import (
    RetentionPolicy,
    plan_pruning,
    protected_checkpoints,
)


def test_a_policy_keeps_its_cadence_its_declared_steps_and_its_final_step() -> None:
    policy = RetentionPolicy(
        final_step=70, keep_every_steps=25, keep_steps=frozenset({30})
    )

    assert [step for step in range(1, 71) if policy.retains(step)] == [
        25,
        30,
        50,
        70,
    ]
    assert [step for step in range(1, 71) if policy.requires(step)] == [
        25,
        30,
        50,
        70,
    ]


def test_a_policy_without_a_cadence_keeps_everything_and_requires_only_the_end() -> (
    None
):
    policy = RetentionPolicy(final_step=70)

    assert all(policy.retains(step) for step in range(1, 71))
    assert [step for step in range(1, 71) if policy.requires(step)] == [70]


def _run(
    root: Path,
    name: str,
    steps: tuple[int, ...],
    *,
    latest: int,
    config: dict[str, object] | None = None,
    resumed_from: Path | None = None,
) -> Path:
    run = root / name
    directory = run / "checkpoints"
    directory.mkdir(parents=True)
    for step in steps:
        (directory / f"step-{step:08d}.pt").write_bytes(b"weights")
    (directory / "latest.json").write_text(
        json.dumps({"global_step": latest, "path": f"step-{latest:08d}.pt"}),
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "resolved_config": {"config": {"steps": 100, **(config or {})}},
                "optimization": {
                    "resumed_from": (
                        str(resumed_from.resolve()) if resumed_from else None
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def _names(paths: tuple[Path, ...]) -> list[str]:
    return [path.name for path in paths]


def test_pruning_keeps_what_the_policy_and_other_readers_need(
    tmp_path: Path,
) -> None:
    trunk = _run(
        tmp_path,
        "trunk",
        (10, 20, 30, 40, 50, 60, 70, 80),
        latest=70,
        config={"keep_every_steps": 40, "keep_steps": [50]},
    )
    _run(
        tmp_path,
        "branch",
        (90,),
        latest=90,
        resumed_from=trunk / "checkpoints/step-00000020.pt",
    )
    write_model_selection(tmp_path, run="trunk", checkpoint="step-00000030.pt")

    plan = plan_pruning(trunk, protected=protected_checkpoints(tmp_path))

    # 40 and 50 are the policy's, 70 is latest, and 80 is above it, where a
    # save still in flight would be.
    assert _names(plan.removable) == ["step-00000010.pt", "step-00000060.pt"]
    assert _names(plan.held) == ["step-00000020.pt", "step-00000030.pt"]
    assert plan.removable_bytes == 2 * len(b"weights")


def test_a_run_that_resumed_from_itself_protects_nothing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _run(
        tmp_path,
        "run",
        (10, 20),
        latest=20,
        config={"keep_every_steps": 1000},
        resumed_from=run / "checkpoints/step-00000010.pt",
    )

    plan = plan_pruning(run, protected=protected_checkpoints(tmp_path))

    assert _names(plan.removable) == ["step-00000010.pt"]


def test_a_cadence_given_to_the_command_never_overrides_the_one_a_run_declares(
    tmp_path: Path,
) -> None:
    undeclared = _run(tmp_path, "undeclared", (10, 20, 30), latest=30)
    declared = _run(
        tmp_path,
        "declared",
        (10, 20, 30),
        latest=30,
        config={"keep_every_steps": 10},
    )

    assert plan_pruning(undeclared, protected=()).removable == ()
    assert _names(
        plan_pruning(undeclared, protected=(), keep_every_steps=20).removable
    ) == ["step-00000010.pt"]
    assert plan_pruning(declared, protected=(), keep_every_steps=20).removable == ()
    assert (
        plan_pruning(
            undeclared, protected=(), keep_every_steps=20, keep_steps=(10,)
        ).removable
        == ()
    )


def test_a_run_without_a_latest_checkpoint_is_left_alone(tmp_path: Path) -> None:
    run = _run(tmp_path, "run", (10, 20), latest=20)
    (run / "checkpoints/latest.json").unlink()

    plan = plan_pruning(run, protected=(), keep_every_steps=1000)

    assert plan.removable == ()
    assert plan.skipped == "no latest checkpoint"


def test_the_prune_command_reports_unless_told_to_delete(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _run(tmp_path, "run", (10, 20, 30), latest=30)
    arguments = ["checkpoints", "prune", "--run-root", str(tmp_path)]
    arguments += ["--keep-every-steps", "20"]

    assert main(arguments) == 0
    assert "would remove 1 checkpoint(s)" in capsys.readouterr().out
    assert (run / "checkpoints/step-00000010.pt").exists()

    assert main([*arguments, "--delete"]) == 0
    assert "removed 1 checkpoint(s)" in capsys.readouterr().out
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == [
        "latest.json",
        "step-00000020.pt",
        "step-00000030.pt",
    ]
