"""The game outcome a value head is trained on, and the head itself."""

from __future__ import annotations

import chess
import pytest
import torch
from tiny_models import tiny_model_config
from torch.nn import functional as F

from anthro_chess.chess import encode_move
from anthro_chess.data import (
    OUTCOME_COUNT,
    DecisionHistory,
    GameEncodingInput,
    SequenceExample,
    collate_packed,
    collate_sequences,
    encode_game,
)
from anthro_chess.data.loading import OUTCOME_DRAW, OUTCOME_LOSS, OUTCOME_WIN
from anthro_chess.models import MoveModel, MoveModelBatch, MoveModelConfig
from anthro_chess.models.batching import OptionalTensor
from anthro_chess.models.move_model import model_identity
from anthro_chess.training.losses import masked_outcome_cross_entropy

_MOVES = ("e2e4", "e7e5", "g1f3", "b8c6")


def _example(result: str, *, game_id: int = 100) -> SequenceExample:
    plies = encode_game(
        GameEncodingInput(
            game_id=game_id,
            ruleset="standard",
            initial_position=chess.STARTING_FEN,
            action_ids=tuple(encode_move(chess.Move.from_uci(m)) for m in _MOVES),
            white_normalized_rating=1500,
            black_normalized_rating=1500,
            time_initial_ms=None,
            time_increment_ms=None,
            clock_remaining_ms=(None,) * len(_MOVES),
        )
    )
    return SequenceExample(
        shard_index=0, game_id=game_id, start_ply=0, plies=plies, result=result
    )


def _outcomes(batch: MoveModelBatch) -> list[int | None]:
    return [
        int(value) if present else None
        for value, present in zip(
            batch.outcome.values[0].tolist(),
            batch.outcome.present[0].tolist(),
            strict=True,
        )
    ]


def test_each_decision_is_labelled_with_the_result_from_its_movers_side() -> None:
    batch = MoveModelBatch.from_sequence_batch(collate_sequences((_example("0-1"),)))

    assert _outcomes(batch) == [OUTCOME_LOSS, OUTCOME_WIN] * 2


def test_a_draw_is_a_draw_for_both_and_an_unknown_result_is_absent() -> None:
    drawn = MoveModelBatch.from_sequence_batch(
        collate_sequences((_example("1/2-1/2"),))
    )
    unknown = MoveModelBatch.from_sequence_batch(collate_sequences((_example("*"),)))

    assert _outcomes(drawn) == [OUTCOME_DRAW] * len(_MOVES)
    assert _outcomes(unknown) == [None] * len(_MOVES)


def test_a_packed_row_labels_each_game_by_its_own_result() -> None:
    examples = (_example("1-0", game_id=1), _example("1/2-1/2", game_id=2))

    batch = MoveModelBatch.from_sequence_batch(
        collate_packed(examples, 2 * len(_MOVES) + 1)
    )

    assert _outcomes(batch) == [
        *[OUTCOME_WIN, OUTCOME_LOSS] * 2,
        *[OUTCOME_DRAW] * len(_MOVES),
        None,
    ]


def test_an_unrecognized_result_is_refused() -> None:
    with pytest.raises(ValueError, match="unrecognized game result"):
        _example("1-1")


def test_a_served_decision_presents_no_outcome() -> None:
    history = DecisionHistory(moves=[chess.Move.from_uci(text) for text in _MOVES])

    batch = MoveModelBatch.from_decision_context(history.context(target_rating=1500))

    assert not batch.outcome.present.any()


def test_the_value_pass_scores_moves_exactly_as_the_ordinary_pass_does() -> None:
    torch.manual_seed(5)
    model = MoveModel(tiny_model_config(value_head=True)).eval()
    batch = MoveModelBatch.from_sequence_batch(collate_sequences((_example("1-0"),)))

    with torch.no_grad():
        action_logits, outcome_logits = model(batch, with_value=True)
        ordinary = model(batch)

    assert torch.equal(action_logits, ordinary)
    assert outcome_logits.shape == (1, len(_MOVES), OUTCOME_COUNT)


def test_a_model_without_a_value_head_refuses_a_value_pass() -> None:
    model = MoveModel(tiny_model_config())
    batch = MoveModelBatch.from_sequence_batch(collate_sequences((_example("1-0"),)))

    with pytest.raises(ValueError, match="no value head"):
        model(batch, with_value=True)


def test_a_model_without_a_value_head_keeps_the_identity_recorded_before_it() -> None:
    plain = model_identity(tiny_model_config())
    valued = model_identity(tiny_model_config(value_head=True))

    assert "value_head" not in plain["config"]  # type: ignore[operator]
    assert "value_loss_weight" not in plain["config"]  # type: ignore[operator]
    rebuilt = MoveModelConfig.model_validate(valued["config"])
    assert rebuilt == tiny_model_config(value_head=True)
    MoveModel(rebuilt).load_state_dict(
        MoveModel(tiny_model_config(value_head=True)).state_dict()
    )


def test_the_outcome_loss_reads_only_known_enabled_decisions() -> None:
    logits = torch.randn(1, 3, OUTCOME_COUNT)
    targets = torch.tensor([[OUTCOME_WIN, OUTCOME_LOSS, OUTCOME_DRAW]])
    outcome = OptionalTensor(targets, torch.tensor([[True, True, False]]))
    enabled = torch.tensor([[True, False, True]])

    loss = masked_outcome_cross_entropy(logits, outcome, enabled)
    nothing_known = masked_outcome_cross_entropy(
        logits, OptionalTensor(targets, torch.zeros_like(enabled)), enabled
    )

    assert loss == pytest.approx(F.cross_entropy(logits[0, :1], targets[0, :1]).item())
    assert nothing_known.item() == 0.0
