import json
from pathlib import Path
from typing import Any

import chess
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from anthro_chess.chess import encode_move
from anthro_chess.config import load_config
from anthro_chess.data import (
    DataLoadingError,
    PrepareConfig,
    SequenceDataLoader,
    SequenceDataset,
    SequenceLoaderConfig,
    prepare_pgn,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
SAMPLE_PGN = REPOSITORY_ROOT / "samples/lichess/standard-export-sample.pgn"
SAMPLE_CONFIG = REPOSITORY_ROOT / "configs/data/lichess-sample.toml"


def test_loads_artifact_written_with_canonical_normalized_schema(
    tmp_path: Path,
) -> None:
    prepared = prepare_pgn(
        SAMPLE_PGN,
        tmp_path / "artifacts",
        load_config(PrepareConfig, path=SAMPLE_CONFIG),
    )
    split = next(name for name, count in prepared.split_counts.items() if count > 0)

    dataset = SequenceDataset.from_parquet(prepared.normalized_path, split=split)

    assert len(dataset) == 1
    assert dataset[0].plies[0].player_rating == 2100


def test_loads_full_games_and_pads_targets_inputs_and_legal_actions(
    tmp_path: Path,
) -> None:
    path = _write_games(
        tmp_path,
        [
            _row(1, ("e2e4", "e7e5", "g1f3")),
            _row(2, ("d2d4",)),
        ],
    )
    loader = SequenceDataLoader.from_parquet(
        path,
        SequenceLoaderConfig(batch_size=2, shuffle=False),
    )

    batch = next(loader)

    assert batch.batch_size == 2
    assert batch.sequence_length == 3
    assert batch.attention_mask == ((True, True, True), (True, False, False))
    assert batch.action_loss_mask == batch.attention_mask
    assert batch.action_targets[0] == _action_ids(("e2e4", "e7e5", "g1f3"))
    assert batch.action_targets[1] == (_action_ids(("d2d4",))[0], 0, 0)
    assert batch.legal_action_ids[1][0]
    assert batch.legal_action_ids[1][1:] == ((), ())
    assert batch.inputs.piece_ids[1][1] == (0,) * 64
    assert batch.inputs.previous_action_id.present[0] == (False, True, True)
    assert batch.inputs.player_rating.present[0] == (True, True, True)
    assert batch.inputs.opponent_rating.values[0] == (1401, 1400, 1401)
    assert batch.game_ids == ((1, 1, 1), (2, 0, 0))
    assert batch.ply_indices == ((0, 1, 2), (0, 0, 0))


def test_fixed_chunks_remain_contiguous_and_tail_padding_is_masked(
    tmp_path: Path,
) -> None:
    path = _write_games(
        tmp_path,
        [_row(7, ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"))],
    )
    dataset = SequenceDataset.from_parquet(
        path,
        split="train",
        chunk_length=2,
    )
    loader = SequenceDataLoader(
        dataset,
        SequenceLoaderConfig(batch_size=3, chunk_length=2, shuffle=False),
    )

    batch = next(loader)

    assert batch.chunk_start_plies == (0, 2, 4)
    assert batch.ply_indices == ((0, 1), (2, 3), (4, 0))
    assert batch.attention_mask == ((True, True), (True, True), (True, False))
    assert batch.inputs.previous_action_id.values[1][0] == _action_ids(("e7e5",))[0]
    assert batch.inputs.previous_action_id.present[1][0] is True


def test_causal_mask_prevents_future_target_attention() -> None:
    dataset = _encoded_dataset()
    loader = SequenceDataLoader(
        dataset,
        SequenceLoaderConfig(batch_size=1, shuffle=False),
    )

    batch = next(loader)

    assert batch.causal_attention_mask == (
        (True, False, False),
        (True, True, False),
        (True, True, True),
    )
    assert batch.inputs.previous_action_id.values[0][1] == batch.action_targets[0][0]
    assert batch.inputs.previous_action_id.values[0][2] == batch.action_targets[0][1]


def test_deterministic_order_changes_by_epoch_and_restores_exact_cursor(
    tmp_path: Path,
) -> None:
    path = _write_games(
        tmp_path,
        [_row(game_id, ("e2e4",)) for game_id in range(1, 9)],
    )
    dataset = SequenceDataset.from_parquet(path, split="train")
    config = SequenceLoaderConfig(batch_size=3, seed="repeatable")
    loader = SequenceDataLoader(dataset, config)

    first_batch = next(loader)
    saved = loader.state()
    expected_remainder = [_batch_game_ids(batch) for batch in loader]

    restored = SequenceDataLoader(dataset, config)
    restored.load_state(json.loads(json.dumps(saved.as_record())))
    assert [_batch_game_ids(batch) for batch in restored] == expected_remainder

    replay = SequenceDataLoader(dataset, config)
    assert _batch_game_ids(next(replay)) == _batch_game_ids(first_batch)
    replay.start_epoch(1)
    epoch_one = [_batch_game_ids(batch) for batch in replay]
    assert epoch_one != [_batch_game_ids(first_batch), *expected_remainder]


def test_rejects_incompatible_state_schema_and_empty_selection(tmp_path: Path) -> None:
    path = _write_games(tmp_path, [_row(1, ("e2e4",))])
    dataset = SequenceDataset.from_parquet(path, split="train")
    loader = SequenceDataLoader(dataset, SequenceLoaderConfig())
    state = loader.state().as_record()

    with pytest.raises(DataLoadingError, match="different sequence data"):
        loader.load_state({**state, "dataset_sha256": "wrong"})
    with pytest.raises(DataLoadingError, match="different loader configuration"):
        other_config = SequenceLoaderConfig(seed="different")
        SequenceDataLoader(dataset, other_config).load_state(state)
    with pytest.raises(DataLoadingError, match="incomplete or unknown"):
        loader.load_state({**state, "extra": 1})
    with pytest.raises(DataLoadingError, match="no normalized games"):
        SequenceDataset.from_parquet(path, split="validation")


def test_rejects_incompatible_normalized_schema(tmp_path: Path) -> None:
    path = _write_games(tmp_path, [_row(1, ("e2e4",), schema_version=99)])

    with pytest.raises(DataLoadingError, match="schema version 99"):
        SequenceDataset.from_parquet(path, split="train")


def test_dataset_identity_covers_content_and_canonical_shard_order(
    tmp_path: Path,
) -> None:
    first_path = _write_games(
        tmp_path,
        [_row(1, ("e2e4",))],
        name="a.parquet",
    )
    second_path = _write_games(
        tmp_path,
        [_row(2, ("d2d4",))],
        name="b.parquet",
    )

    ordered = SequenceDataset.from_parquet(
        [first_path, second_path],
        split="train",
    )
    reversed_input = SequenceDataset.from_parquet(
        [second_path, first_path],
        split="train",
    )
    changed = _write_games(
        tmp_path,
        [_row(1, ("c2c4",))],
        name="changed.parquet",
    )

    assert [example.game_id for example in ordered] == [1, 2]
    assert ordered.identity_sha256 == reversed_input.identity_sha256
    assert (
        ordered.identity_sha256
        != SequenceDataset.from_parquet(changed, split="train").identity_sha256
    )


def _encoded_dataset() -> SequenceDataset:
    from anthro_chess.data import GameEncodingInput, SequenceExample, encode_game

    game = GameEncodingInput(
        game_id=99,
        ruleset="standard",
        initial_position=chess.STARTING_FEN,
        action_ids=_action_ids(("e2e4", "e7e5", "g1f3")),
        white_normalized_rating=None,
        black_normalized_rating=None,
        time_initial_ms=None,
        time_increment_ms=None,
        clock_remaining_ms=(None, None, None),
    )
    plies = encode_game(game)
    return SequenceDataset(
        [SequenceExample(0, game.game_id, 0, plies)],
        identity_sha256="test",
    )


def _write_games(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    name: str = "games.parquet",
) -> Path:
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(
    game_id: int,
    moves: tuple[str, ...],
    *,
    schema_version: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "game_id": game_id,
        "ruleset": "standard",
        "initial_position": chess.STARTING_FEN,
        "action_ids": list(_action_ids(moves)),
        "white_normalized_rating": 1400,
        "black_normalized_rating": 1401,
        "time_initial_ms": 60_000,
        "time_increment_ms": 1_000,
        "clock_remaining_ms": [59_000] * len(moves),
        "split": "train",
    }


def _action_ids(moves: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(encode_move(chess.Move.from_uci(move)) for move in moves)


def _batch_game_ids(batch: object) -> tuple[int, ...]:
    from anthro_chess.data import SequenceBatch

    assert isinstance(batch, SequenceBatch)
    return tuple(row[0] for row in batch.game_ids)
