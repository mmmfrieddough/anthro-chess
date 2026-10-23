"""The move-time head, the distribution it writes, and the times it trains on."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import chess
import pytest
import torch
from tiny_models import tiny_model_config

from anthro_chess.chess import encode_move
from anthro_chess.data import (
    GameEncodingInput,
    SequenceExample,
    collate_sequences,
    encode_game,
)
from anthro_chess.models import (
    MoveModel,
    MoveModelBatch,
    MoveTimeDistribution,
    move_time_targets,
)
from anthro_chess.models.move_model import model_identity
from anthro_chess.models.timing import MOVE_TIME_COMPONENTS, MOVE_TIME_RESOLUTION_MS
from anthro_chess.training.losses import masked_move_time_loss

_MOVES = ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4")
_INITIAL_MS = 60_000
_INCREMENT_MS = 1_000
#: Each player's clock after each move, increment included. The first move of
#: each side runs no clock; after that white spends four seconds then five,
#: and black one.
_CLOCKS_MS = (60_030, 60_030, 57_030, 60_030, 53_030)


def _batch(
    *,
    initial_ms: int | None = _INITIAL_MS,
    increment_ms: int | None = _INCREMENT_MS,
    clocks_ms: Sequence[int | None] = _CLOCKS_MS,
) -> MoveModelBatch:
    plies = encode_game(
        GameEncodingInput(
            game_id=7,
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
    return MoveModelBatch.from_sequence_batch(
        collate_sequences(
            (SequenceExample(shard_index=0, game_id=7, start_ply=0, plies=plies),)
        )
    )


def _mixture(
    locations: Sequence[float], log_scales: Sequence[float]
) -> MoveTimeDistribution:
    weights = torch.zeros(len(locations))
    return MoveTimeDistribution(
        logits=weights,
        locations=torch.tensor(locations),
        log_scales=torch.tensor(log_scales),
    )


def test_the_recorded_intervals_hold_the_whole_distribution() -> None:
    distribution = _mixture((0.5, 7.5, 10.0), (0.3, 0.0, -3.0))
    times = torch.arange(0, 3_000_000, MOVE_TIME_RESOLUTION_MS)

    mass = distribution.log_prob(times.unsqueeze(-1)).exp().sum()

    assert mass.item() == pytest.approx(1.0, abs=1e-3)


def test_a_long_think_is_scored_as_finely_as_a_quick_move() -> None:
    location, log_scale = 13.0, -2.0
    distribution = _mixture((location,), (log_scale,))
    time_ms = 450_000.0

    def cdf(edge: float) -> float:
        logged = math.log1p(edge)
        return 1.0 / (1.0 + math.exp(-(logged - location) / math.exp(log_scale)))

    half = MOVE_TIME_RESOLUTION_MS / 2
    expected = math.log(cdf(time_ms + half) - cdf(time_ms - half))

    measured = distribution.log_prob(torch.tensor(time_ms)).item()

    assert measured == pytest.approx(expected, abs=1e-3)


def test_a_sample_falls_where_the_distribution_puts_its_mass() -> None:
    distribution = MoveTimeDistribution(
        logits=torch.tensor([[0.0, -20.0]]).expand(4_000, 2),
        locations=torch.tensor([[math.log1p(2_000.0), 0.0]]).expand(4_000, 2),
        log_scales=torch.tensor([[-1.0, 0.0]]).expand(4_000, 2),
    )

    sampled = distribution.sample(torch.Generator().manual_seed(3))

    assert sampled.dtype == torch.long
    assert sampled.median().item() == pytest.approx(2_000, rel=0.05)
    assert (sampled >= 0).all()


def test_a_move_time_is_the_clock_it_spent_and_the_first_two_plies_have_none() -> None:
    targets = move_time_targets(_batch())

    assert targets.present.tolist() == [[False, False, True, True, True]]
    assert targets.values[0, 2:].tolist() == [4_000, 1_000, 5_000]


def test_an_untimed_game_has_no_move_times() -> None:
    targets = move_time_targets(
        _batch(initial_ms=None, increment_ms=None, clocks_ms=(None,) * len(_MOVES))
    )

    assert not targets.present.any()


def test_a_batch_without_a_timed_move_contributes_no_timing_loss() -> None:
    batch = _batch(initial_ms=None, increment_ms=None, clocks_ms=(None,) * 5)
    parameters = torch.zeros((*batch.action_targets.shape, 3 * MOVE_TIME_COMPONENTS))

    loss = masked_move_time_loss(
        parameters, move_time_targets(batch), batch.action_loss_mask
    )

    assert loss.item() == 0.0


def test_the_time_head_reads_the_chosen_move_and_trains_the_trunk() -> None:
    torch.manual_seed(5)
    model = MoveModel(tiny_model_config(clock_inputs=True, time_head=True))
    batch = _batch()

    logits, times = model(batch, with_time=True)
    loss = masked_move_time_loss(
        times, move_time_targets(batch), batch.action_loss_mask
    )
    loss.backward()

    assert logits.shape == (*batch.action_targets.shape, logits.shape[-1])
    assert times.shape == (*batch.action_targets.shape, 3 * MOVE_TIME_COMPONENTS)
    assert torch.isfinite(loss)
    trunk = model.square_encoder.piece_projection.weight.grad
    assert trunk is not None and trunk.abs().sum() > 0

    model.eval()
    squares = model.encode(batch, batch.decision_columns)
    black = batch.inputs.side_to_move.bool()
    other = torch.full_like(
        batch.action_targets, encode_move(chess.Move.from_uci("a2a3"))
    )
    assert model.time_head is not None
    assert not torch.allclose(
        model.time_head(squares, black, batch.action_targets),
        model.time_head(squares, black, other),
    )


def test_a_model_without_the_head_keeps_the_identity_it_always_had() -> None:
    without = model_identity(tiny_model_config(clock_inputs=True))
    with_head = model_identity(tiny_model_config(clock_inputs=True, time_head=True))

    assert without["timing_head"] is False
    assert "time_head" not in cast(dict[str, object], without["config"])
    assert "time_loss_weight" not in cast(dict[str, object], without["config"])
    assert cast(dict[str, object], with_head["config"])["time_loss_weight"] == 1.0
    assert with_head["timing_head"] == {
        "distribution": "logistic-mixture-over-log1p-ms",
        "components": MOVE_TIME_COMPONENTS,
        "resolution_ms": MOVE_TIME_RESOLUTION_MS,
    }


def test_a_model_without_the_head_refuses_to_write_time() -> None:
    model = MoveModel(tiny_model_config())

    with pytest.raises(ValueError, match="no time head"):
        model(_batch(), with_time=True)
