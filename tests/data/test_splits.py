"""Contract tests for the deterministic three-way game-level split."""

from pathlib import Path

import pytest

from anthro_chess.config import ConfigError, load_config
from anthro_chess.data import PrepareConfig, SplitConfig, prepare_pgn
from anthro_chess.data.schema import SPLIT_NAMES, split_name

REPOSITORY_ROOT = Path(__file__).parents[2]
SAMPLE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"

_SEED = "contract-seed"
_FRACTIONS = {"validation_fraction": 0.1, "test_fraction": 0.1}


def _assign(game_id: int, **fractions: float) -> str:
    return split_name(game_id, seed=_SEED, **{**_FRACTIONS, **fractions})


def test_assignment_depends_only_on_seed_and_game_id() -> None:
    game_ids = range(2000)

    assignments = [_assign(game_id) for game_id in game_ids]

    assert set(assignments) == set(SPLIT_NAMES)
    assert assignments == [_assign(game_id) for game_id in game_ids]
    assert [
        split_name(game_id, seed="other", **_FRACTIONS) for game_id in game_ids
    ] != (assignments)


def test_growing_a_corpus_never_moves_a_game_between_splits() -> None:
    """Adding games or refiltering must not reassign an existing game.

    This is what lets a frozen test pool stay safe as the corpus grows: a
    game held out today cannot appear in a later training selection.
    """

    original = {game_id: _assign(game_id) for game_id in range(500)}

    grown = {game_id: _assign(game_id) for game_id in range(5000)}

    assert all(grown[game_id] == split for game_id, split in original.items())


def test_test_membership_survives_a_changed_validation_fraction() -> None:
    """The test split claims the lowest hash range so it stays stable."""

    game_ids = range(3000)
    before = {game_id for game_id in game_ids if _assign(game_id) == "test"}

    after = {
        game_id
        for game_id in game_ids
        if _assign(game_id, validation_fraction=0.3) == "test"
    }

    assert before == after
    assert before


def test_zero_test_fraction_reproduces_the_two_way_partition() -> None:
    game_ids = range(1000)

    assignments = {game_id: _assign(game_id, test_fraction=0.0) for game_id in game_ids}

    assert "test" not in set(assignments.values())
    assert set(assignments.values()) == {"train", "validation"}


def test_fractions_must_leave_a_nonempty_train_split() -> None:
    with pytest.raises(ValueError, match="nonempty train split"):
        SplitConfig(validation_fraction=0.6, test_fraction=0.4)

    assert SplitConfig(validation_fraction=0.6, test_fraction=0.39)


def test_prepared_manifest_records_the_three_way_recipe_and_counts(
    tmp_path: Path,
) -> None:
    import json

    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=("split.test_fraction=0.3",),
    )

    result = prepare_pgn(
        REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn",
        tmp_path / "artifacts",
        resolved,
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["split"]["algorithm"] == "sha256-threshold-v2"
    assert manifest["split"]["test_fraction"] == 0.3
    assert set(manifest["split"]["counts"]) == set(SPLIT_NAMES)
    assert set(result.split_counts) == set(SPLIT_NAMES)


def test_nonempty_requirement_ignores_splits_that_were_not_requested(
    tmp_path: Path,
) -> None:
    """A zero test fraction must not fail the nonempty check it never asked for."""

    resolved = load_config(
        PrepareConfig,
        path=SAMPLE_CONFIG,
        overrides=(
            "split.validation_fraction=0.0",
            "split.test_fraction=0.0",
            "split.require_nonempty=true",
        ),
    )

    result = prepare_pgn(
        REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn",
        tmp_path / "artifacts",
        resolved,
    )

    assert result.split_counts["train"] == 1
    assert result.split_counts["test"] == 0


def test_training_configuration_refuses_the_held_out_test_split() -> None:
    from anthro_chess.training import TrainingConfig

    selection = {
        "normalized": "artifacts/x/normalized",
        "manifest": "artifacts/x/manifests/manifest.json",
    }

    with pytest.raises(ConfigError, match="must not use the held-out test split"):
        load_config(
            TrainingConfig,
            overrides=(
                f'train.normalized="{selection["normalized"]}"',
                f'train.manifest="{selection["manifest"]}"',
                'train.loader.split="test"',
            ),
        )
