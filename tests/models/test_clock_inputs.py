"""The time context a clock-reading move model is given, and what it is not."""

from __future__ import annotations

import math
from collections.abc import Sequence

import chess
import pytest
import torch
from tiny_models import tiny_model_config
from torch import nn

from anthro_chess.chess import encode_move
from anthro_chess.data import (
    DecisionHistory,
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    collate_packed,
    collate_sequences,
    encode_game,
)
from anthro_chess.models import MoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.models.move_model import model_identity

_MOVES = ("e2e4", "e7e5", "g1f3", "b8c6")
_INITIAL_MS = 60_000
_INCREMENT_MS = 1_000
#: Each player's clock after each move, increment included: white spends two
#: seconds then four, black three then one.
_CLOCKS_MS = (59_000, 58_000, 56_000, 58_000)


def _config(
    *, clock_inputs: bool = True, clock_dropout: float = 0.0
) -> MoveModelConfig:
    return tiny_model_config(clock_inputs=clock_inputs, clock_dropout=clock_dropout)


def _example(
    *,
    game_id: int = 100,
    initial_ms: int | None = _INITIAL_MS,
    increment_ms: int | None = _INCREMENT_MS,
    clocks_ms: Sequence[int | None] = _CLOCKS_MS,
) -> SequenceExample:
    plies = encode_game(
        GameEncodingInput(
            game_id=game_id,
            ruleset="standard",
            initial_position=chess.STARTING_FEN,
            action_ids=tuple(encode_move(chess.Move.from_uci(m)) for m in _MOVES),
            white_normalized_rating=1500,
            black_normalized_rating=1500,
            time_initial_ms=initial_ms,
            time_increment_ms=increment_ms,
            clock_remaining_ms=tuple(clocks_ms),
        )
    )
    return SequenceExample(shard_index=0, game_id=game_id, start_ply=0, plies=plies)


def _batch(*examples: SequenceExample) -> MoveModelBatch:
    return MoveModelBatch.from_sequence_batch(
        collate_sequences(examples or (_example(),))
    )


def _read_features(model: MoveModel, batch: MoveModelBatch) -> torch.Tensor:
    """Return what the clock embedding's projection was handed, per decision."""

    embedding = model.square_encoder.clock_embedding
    assert embedding is not None
    seen: list[torch.Tensor] = []
    first = embedding.projection[0]
    assert isinstance(first, nn.Linear)
    handle = first.register_forward_hook(
        lambda _module, args, _output: seen.append(args[0].detach())
    )
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    return seen[0]


def _seconds(milliseconds: int) -> float:
    return math.log1p(milliseconds / 1000)


def test_each_decision_reads_its_own_clocks_and_both_last_move_times() -> None:
    model = MoveModel(_config()).eval()

    features = _read_features(model, _batch())[0]
    values, present = features[:, :6], features[:, 6:]

    # Initial, increment, mover's clock, opponent's clock, mover's last move,
    # opponent's last move.
    assert present.tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]
    expected = [
        (60_000, 1_000, 60_000, 60_000, 0, 0),
        (60_000, 1_000, 60_000, 59_000, 0, 2_000),
        (60_000, 1_000, 59_000, 58_000, 2_000, 3_000),
        (60_000, 1_000, 58_000, 56_000, 3_000, 4_000),
    ]
    assert values.flatten().tolist() == pytest.approx(
        [_seconds(value) for row in expected for value in row]
    )


def test_a_missing_clock_is_an_absence_rather_than_a_zero_clock() -> None:
    model = MoveModel(_config()).eval()
    clocks = (59_000, None, 56_000, 58_000)

    features = _read_features(model, _batch(_example(clocks_ms=clocks)))[0]

    # Black's clock after its first move is unknown, so the opponent's clock at
    # the third decision and both times that difference it go absent with it.
    assert features[2, 6:].tolist() == [1, 1, 1, 0, 1, 0]
    assert features[3, 6:].tolist() == [1, 1, 0, 1, 0, 1]
    assert features[2, 3].item() == 0.0


def test_an_untimed_game_presents_no_time_context_at_all() -> None:
    model = MoveModel(_config()).eval()

    features = _read_features(
        model,
        _batch(_example(initial_ms=None, increment_ms=None, clocks_ms=(None,) * 4)),
    )[0]

    assert not features.any()


def test_a_packed_decision_does_not_difference_the_clocks_of_the_game_before_it() -> (
    None
):
    model = MoveModel(_config()).eval()
    packed = MoveModelBatch.from_sequence_batch(
        collate_packed((_example(game_id=1), _example(game_id=2)), 2 * len(_MOVES))
    )

    features = _read_features(model, packed)[0]

    assert torch.equal(features[len(_MOVES) :], features[: len(_MOVES)])


def test_the_clock_changes_what_a_clock_reading_model_predicts() -> None:
    torch.manual_seed(5)
    model = MoveModel(_config()).eval()
    blind = MoveModel(_config(clock_inputs=False)).eval()
    hurried = _batch(_example(clocks_ms=(59_000, 58_000, 2_000, 58_000)))

    with torch.no_grad():
        assert not torch.equal(model(_batch())[0, 3], model(hurried)[0, 3])
        assert torch.equal(blind(_batch())[0, 3], blind(hurried)[0, 3])


def test_clock_dropout_hides_the_whole_context_and_only_while_training() -> None:
    model = MoveModel(_config(clock_dropout=1.0))

    assert not _read_features(model.train(), _batch())[..., 6:].any()
    assert _read_features(model.eval(), _batch())[..., 6:].any()


def test_training_draws_every_partial_context_a_decision_can_be_served_with() -> None:
    torch.manual_seed(3)
    model = MoveModel(_config(clock_dropout=0.5)).train()
    examples = tuple(_example(game_id=game) for game in range(64))

    presence = _read_features(model, _batch(*examples))[..., 6:]

    # The last decision of each game has all six fields to lose.
    drawn = {tuple(int(flag) for flag in row) for row in presence[:, -1].tolist()}
    assert drawn == {
        (1, 1, 1, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (1, 1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    }


def test_a_served_decision_presents_its_time_context_as_absent() -> None:
    history = DecisionHistory(moves=[chess.Move.from_uci(text) for text in _MOVES])

    batch = MoveModelBatch.from_decision_context(history.context(target_rating=1500))

    inputs = batch.inputs
    for value in (
        inputs.time_initial_ms,
        inputs.time_increment_ms,
        inputs.player_clock_ms,
        inputs.opponent_clock_ms,
    ):
        assert not value.present.any()


def test_a_model_reading_no_clock_keeps_the_identity_recorded_before_clocks() -> None:
    blind = model_identity(_config(clock_inputs=False))
    timed = model_identity(_config())

    assert "clock_inputs" not in blind["config"]  # type: ignore[operator]
    assert "clock_dropout" not in blind["config"]  # type: ignore[operator]
    assert blind["timing_inputs"] is False
    assert timed["timing_inputs"] is True
    assert MoveModelConfig.model_validate(timed["config"]) == _config()


def test_the_loader_supplies_the_time_context_the_model_reads() -> None:
    loader_batch: SequenceBatch = collate_sequences((_example(),))

    batch = MoveModelBatch.from_sequence_batch(loader_batch)

    assert batch.inputs.player_clock_ms.values[0].tolist() == [
        60_000,
        60_000,
        59_000,
        58_000,
    ]
    assert batch.inputs.opponent_clock_ms.values[0].tolist() == [
        60_000,
        59_000,
        58_000,
        56_000,
    ]
    assert batch.inputs.time_increment_ms.present.all()
