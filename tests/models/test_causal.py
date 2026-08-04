from __future__ import annotations

import math
from typing import cast

import chess
import pytest
import torch
import torch._dynamo
from torch import nn
from torch._dynamo.testing import CompileCounter

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    action_vocabulary_identity,
    decode_move,
    encode_move,
    legal_action_ids,
)
from anthro_chess.data import (
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    collate_sequences,
    encode_game,
    encoding_identity,
)
from anthro_chess.models import (
    CausalMoveModel,
    MoveModelBatch,
    MoveModelConfig,
)
from anthro_chess.models.causal import TransformerBlock
from anthro_chess.training import masked_action_cross_entropy

from accelerators import training_accelerator_parameters


def test_tensor_boundary_preserves_board_target_context_and_padding() -> None:
    loader_batch = _sequence_batch(
        ("e2e4", "e7e5", "g1f3"),
        ("d2d4",),
        white_rating=1500,
    )

    batch = MoveModelBatch.from_sequence_batch(loader_batch)
    initial_board = chess.Board()
    first_action = encode_move(chess.Move.from_uci("e2e4"))
    second_action = encode_move(chess.Move.from_uci("e7e5"))

    assert batch.inputs.piece_ids.shape == (2, 3, 64)
    assert batch.inputs.piece_ids.dtype == torch.long
    assert batch.inputs.piece_ids[0, 0].tolist() == list(
        loader_batch.inputs.piece_ids[0][0]
    )
    assert batch.inputs.side_to_move[0].tolist() == [0, 1, 0]
    assert batch.inputs.castling_rights[0].tolist() == [15, 15, 15]
    assert batch.inputs.en_passant_square.present[0].tolist() == [
        False,
        True,
        True,
    ]
    assert batch.inputs.en_passant_square.values[0].tolist() == [
        0,
        chess.E3,
        chess.E6,
    ]
    assert batch.inputs.halfmove_clock[0].tolist() == [0, 0, 0]
    assert batch.inputs.fullmove_number[0].tolist() == [1, 1, 2]
    assert batch.inputs.previous_action_id.present[0].tolist() == [
        False,
        True,
        True,
    ]
    assert batch.inputs.previous_action_id.values[0].tolist() == [
        0,
        first_action,
        second_action,
    ]
    assert batch.inputs.target_rating.present[0].tolist() == [True, False, True]
    assert batch.inputs.target_rating.values[0].tolist() == [1500, 0, 1500]
    assert decode_move(int(batch.action_targets[0, 0].item())).uci() == "e2e4"
    assert batch.legal_action_ids is not None
    assert tuple(batch.legal_action_ids[0][0]) == legal_action_ids(
        initial_board,
        include_resignation=True,
        include_draw_claim=True,
    )
    assert batch.game_ids[0].tolist() == [100, 100, 100]
    assert batch.ply_indices[0].tolist() == [0, 1, 2]
    assert batch.attention_mask[1].tolist() == [True, False, False]
    assert batch.action_loss_mask.tolist() == [
        [True, True, True],
        [True, False, False],
    ]


def test_tensor_boundary_preserves_unsigned_normalized_game_ids() -> None:
    game_id = 2**63 + 7

    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4",), game_id_base=game_id)
    )

    assert batch.game_ids.dtype == torch.uint64
    assert int(batch.game_ids[0, 0].item()) == game_id


def test_forward_is_cpu_only_and_action_vocabulary_compatible() -> None:
    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(
            ("e2e4", "e7e5", "g1f3"),
            ("d2d4",),
        )
    )
    model = CausalMoveModel(_tiny_config())

    logits = model(batch)

    assert logits.shape == (2, 3, ACTION_VOCABULARY_SIZE)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    assert model.identity()["version"] == 5
    assert model.identity()["action_vocabulary"] == action_vocabulary_identity()
    assert model.identity()["encoding"] == encoding_identity()
    assert (
        model.identity()["rating_conditioning"] == "post-transformer-feature-modulation"
    )
    assert model.identity()["timing_inputs"] is False
    assert model.identity()["timing_head"] is False


def test_padding_cannot_change_what_a_real_timestep_predicts() -> None:
    """What replaces the key padding mask, and what makes padded logits ignorable.

    A shorter history batched beside a longer one must see exactly what it
    would have seen alone.
    """

    torch.manual_seed(19)
    alone = MoveModelBatch.from_sequence_batch(_sequence_batch(("d2d4",)))
    padded = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("d2d4",), ("e2e4", "e7e5", "g1f3"))
    )
    model = CausalMoveModel(_tiny_config()).eval()

    with torch.no_grad():
        alone_logits = model(alone)
        padded_logits = model(padded)

    torch.testing.assert_close(
        padded_logits[0, :1],
        alone_logits[0, :1],
        rtol=0.0,
        atol=1e-6,
    )
    # The padded columns are not zeroed any more, and nothing reads them.
    assert torch.isfinite(padded_logits).all()
    assert torch.count_nonzero(padded_logits[0, 1:]) > 0


def test_the_position_table_is_sized_from_the_declared_context_and_never_saved() -> (
    None
):
    """It is a function of the configuration, so it is not state to restore.

    Causality used to be a second table beside it, sized the same way and
    quadratic where this one is linear. Attention states it as a flag now, so
    the only derived table left is this one.
    """

    config = _tiny_config()
    model = CausalMoveModel(config)
    declared = config.maximum_context_plies

    assert model.position_table.shape == (declared, config.model_dim)
    assert "position_table" not in set(model.state_dict())
    # The weights outlive the bound, so a differently bounded model loads them.
    CausalMoveModel(_tiny_config(maximum_context_plies=32)).load_state_dict(
        model.state_dict()
    )


def test_the_block_stack_computes_what_the_encoder_wrapper_it_replaced_did() -> None:
    """The function stayed where it was; the parameter names and the mask moved.

    The wrapper had to be handed a materialized triangular mask *and* the
    causal flag, because it reads the flag as a hint accompanying a mask rather
    than as a substitute for one. The blocks ask attention for the causal form
    directly and arrive at the same numbers.

    Two layers because that is what every training configuration runs, and
    because a stack is where the residual order and the trailing normalization
    could disagree with the wrapper while one block alone still matched.
    """

    torch.manual_seed(23)
    config = _tiny_config(transformer_layers=2)
    model = CausalMoveModel(config).eval()
    wrapper = _encoder_wrapper(config, model).eval()
    hidden = torch.randn(3, 11, config.model_dim)
    causal = nn.Transformer.generate_square_subsequent_mask(hidden.shape[1])

    with torch.no_grad():
        explicit = hidden
        for block in model.transformer_blocks:
            explicit = block(explicit)
        explicit = model.transformer_norm(explicit)
        wrapped = wrapper(hidden, mask=causal, is_causal=True)

    torch.testing.assert_close(explicit, wrapped, rtol=1e-5, atol=1e-5)


def test_every_layer_is_drawn_on_its_own_rather_than_copied() -> None:
    """The one thing about a fresh model this change did move.

    ``nn.TransformerEncoder`` built its stack by deep-copying one prototype
    layer, so the configured two-layer model began training with two identical
    layers. Explicit construction draws each block, and a later collapse back
    to a cloned prototype would be a silent return to a degenerate start.
    """

    torch.manual_seed(31)
    model = CausalMoveModel(_tiny_config(transformer_layers=2))

    first, second = (
        cast(TransformerBlock, block).attention.qkv_projection.weight
        for block in model.transformer_blocks
    )

    assert not torch.equal(first, second)


def test_the_dropout_setting_reaches_attention_and_the_block_around_it() -> None:
    """Four framework-owned dropout sites became four written-out ones.

    Every configuration sets this to zero, so a site wired to nothing would be
    invisible until the first run that turns the dial up. Attention is checked
    on its own because its dropout is the one inside the fused attention call
    rather than a module the block composes.
    """

    torch.manual_seed(37)
    config = _tiny_config(dropout=0.5)
    block = cast(TransformerBlock, CausalMoveModel(config).transformer_blocks[0])
    hidden = torch.randn(2, 7, config.model_dim)

    with torch.no_grad():
        attention = [block.attention.train()(hidden) for _ in range(2)]
        trained = [block.train()(hidden) for _ in range(2)]
        evaluated = [block.eval()(hidden) for _ in range(2)]

    assert not torch.equal(attention[0], attention[1])
    assert not torch.equal(trained[0], trained[1])
    torch.testing.assert_close(evaluated[0], evaluated[1], rtol=0.0, atol=0.0)


def test_the_forward_pass_compiles_whole_and_once_across_a_run_of_widths() -> None:
    """Length buckets vary the padded width, and none of them is a recompile.

    ``fullgraph`` is what makes a graph break an error here rather than a
    silently slower run. The remaining Python-level guard is the batch's own
    ``chunk_start_plies``, which is `#275`; every row below starts at ply zero,
    as a full-game selection feeds them, so width is what varies.
    """

    torch._dynamo.reset()
    counter = CompileCounter()
    model = CausalMoveModel(_tiny_config(maximum_context_plies=64)).eval()
    compiled = torch.compile(model, backend=counter, fullgraph=True, dynamic=True)

    with torch.no_grad():
        logits = [compiled(_batch_of_width(width)) for width in (4, 6, 8, 12, 20, 30)]

    assert counter.frame_count == 1
    assert [tuple(value.shape) for value in logits] == [
        (1, width, ACTION_VOCABULARY_SIZE) for width in (4, 6, 8, 12, 20, 30)
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("backend", training_accelerator_parameters())
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_the_compiled_forward_pass_agrees_with_the_eager_one(
    backend: str,
    mode: str,
) -> None:
    """What `fullgraph` and CUDA graphs are worth is only worth having if equal.

    ``reduce-overhead`` replays a captured graph instead of reissuing the
    step's kernels, which is where the whole-graph property pays; it is also
    the mode most able to return something subtly wrong, so it is compared
    against eager rather than merely run.
    """

    torch._dynamo.reset()
    torch.manual_seed(29)
    model = CausalMoveModel(_tiny_config(maximum_context_plies=64)).to(backend).eval()
    batch = _batch_of_width(12, device=backend)
    compiled = torch.compile(model, fullgraph=True, mode=mode)

    with torch.no_grad():
        expected = model(batch)
        # Twice: a captured graph is replayed on the second call, not the first.
        compiled(batch)
        actual = compiled(batch)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("start_ply", [0, 2], ids=["from-ply-zero", "chunk-past-zero"])
def test_position_features_are_read_at_each_timestep_own_ply_index(
    start_ply: int,
) -> None:
    """A chunked game's indices run past its own width, and the table reaches."""

    config = _tiny_config()
    model = CausalMoveModel(config)
    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3", "b8c6"), start_ply=start_ply)
    )

    features = model.position_table[batch.ply_indices]

    assert int(batch.ply_indices[0, 0].item()) == start_ply
    dimension = config.model_dim
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    angles = batch.ply_indices.float().unsqueeze(-1) * frequencies
    expected = torch.zeros((*batch.ply_indices.shape, dimension))
    expected[..., 0::2] = torch.sin(angles)
    expected[..., 1::2] = torch.cos(angles)

    torch.testing.assert_close(features, expected)


def test_position_features_do_not_narrow_with_the_forward_pass() -> None:
    """The table is the model's own, not the width the forward pass runs at.

    bfloat16 carries eight mantissa bits, so ply 257 and ply 258 are the same
    number in it. Building the table at whatever width attention happened to be
    running under collapsed one onto the other.
    """

    model = CausalMoveModel(_tiny_config(maximum_context_plies=512))

    assert model.position_table.dtype == torch.float32
    assert not torch.equal(model.position_table[257], model.position_table[258])


def test_a_batch_reaching_past_the_declared_context_is_refused() -> None:
    """A late chunk is measured by the ply it reaches, not by how wide it is."""

    moves = ("e2e4", "e7e5", "g1f3", "b8c6")
    model = CausalMoveModel(_tiny_config(maximum_context_plies=3))
    whole = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    tail = MoveModelBatch.from_sequence_batch(_sequence_batch(moves, start_ply=2))

    assert tail.ply_indices.shape[1] < 3
    for batch in (whole, tail):
        with pytest.raises(ValueError, match="ply index 3, past the 3 plies"):
            model(batch)


def test_the_identity_carries_every_value_needed_to_rebuild_the_model() -> None:
    """A checkpoint is rebuilt from its identity, so a gap becomes a default."""

    config = _tiny_config()
    config_record = CausalMoveModel(config).identity()["config"]

    assert config_record == config.model_dump(mode="json")
    assert MoveModelConfig.model_validate(config_record) == config
    assert (
        CausalMoveModel(_tiny_config(maximum_context_plies=32)).identity()
        != CausalMoveModel(config).identity()
    )


def test_future_context_does_not_change_earlier_predictions() -> None:
    torch.manual_seed(3)
    original = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3"))
    )
    changed = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "d2d4"))
    )
    changed.inputs.piece_ids[0, 2].zero_()
    model = CausalMoveModel(_tiny_config()).eval()

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    torch.testing.assert_close(
        original_logits[:, :2],
        changed_logits[:, :2],
        rtol=0.0,
        atol=1e-6,
    )
    assert not torch.equal(original_logits[:, 2], changed_logits[:, 2])


def test_rating_changes_decision_without_changing_encoded_history() -> None:
    torch.manual_seed(7)
    unrated = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3"))
    )
    rated = MoveModelBatch.from_sequence_batch(
        _sequence_batch(
            ("e2e4", "e7e5", "g1f3"),
            white_rating=2000,
            black_rating=1800,
        )
    )
    model = CausalMoveModel(_tiny_config()).eval()

    with torch.no_grad():
        unrated_history = model.encode_history(unrated)
        rated_history = model.encode_history(rated)
        unrated_logits = model(unrated)
        rated_logits = model(rated)

    torch.testing.assert_close(unrated_history, rated_history, rtol=0.0, atol=0.0)
    assert not torch.equal(unrated_logits, rated_logits)


@pytest.mark.parametrize(
    "moves",
    [
        ("e2e4",),
        ("e2e4", "e7e5", "g1f3"),
    ],
    ids=["single-timestep", "short-causal-sequence"],
)
def test_ordinary_model_and_loss_path_overfits_a_fixed_tiny_sample(
    moves: tuple[str, ...],
) -> None:
    torch.manual_seed(11)
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    assert not torch.any(batch.inputs.target_rating.present)
    model = CausalMoveModel(_tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        initial_loss_tensor = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        )
        initial_loss = initial_loss_tensor.item()

    for _ in range(60):
        optimizer.zero_grad()
        loss = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = masked_action_cross_entropy(
            model(batch),
            batch.action_targets,
            batch.action_loss_mask,
        ).item()

    expected_untrained_loss = math.log(ACTION_VOCABULARY_SIZE)
    assert torch.isfinite(initial_loss_tensor)
    assert abs(initial_loss - expected_untrained_loss) < 2.0
    assert final_loss < initial_loss * 0.2


def test_model_config_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MoveModelConfig(model_dim=18, attention_heads=4)
    with pytest.raises(ValueError, match="even"):
        MoveModelConfig(model_dim=15, attention_heads=3)


def _sequence_batch(
    *move_lines: tuple[str, ...],
    white_rating: int | None = None,
    black_rating: int | None = None,
    game_id_base: int = 100,
    start_ply: int = 0,
) -> SequenceBatch:
    examples = []
    for game_offset, moves in enumerate(move_lines):
        board = chess.Board()
        action_ids = []
        for move_text in moves:
            move = chess.Move.from_uci(move_text)
            assert move in board.legal_moves
            action_ids.append(encode_move(move))
            board.push(move)
        plies = encode_game(
            GameEncodingInput(
                game_id=game_id_base + game_offset,
                ruleset="standard",
                initial_position=chess.STARTING_FEN,
                action_ids=tuple(action_ids),
                white_normalized_rating=white_rating,
                black_normalized_rating=black_rating,
                time_initial_ms=None,
                time_increment_ms=None,
                clock_remaining_ms=tuple(None for _ in action_ids),
            )
        )
        examples.append(
            SequenceExample(
                shard_index=0,
                game_id=game_id_base + game_offset,
                start_ply=start_ply,
                plies=plies[start_ply:],
            )
        )
    return collate_sequences(examples)


def _batch_of_width(
    plies: int,
    *,
    device: str | None = None,
) -> MoveModelBatch:
    """Return one full game of ``plies`` plies, played by an arbitrary rule.

    Which moves they are does not matter to a shape or a graph, and picking
    them by rule is what lets a caller ask for a width rather than write one.
    """

    board = chess.Board()
    moves = []
    for _ in range(plies):
        move = next(iter(board.legal_moves))
        moves.append(move.uci())
        board.push(move)
    return MoveModelBatch.from_sequence_batch(
        _sequence_batch(tuple(moves)), device=device
    )


def _encoder_wrapper(
    config: MoveModelConfig,
    model: CausalMoveModel,
) -> nn.TransformerEncoder:
    """Return the replaced wrapper, holding the explicit blocks' own weights.

    This characterizes one migration rather than stating an invariant, and it
    is the only thing in the project still naming the framework's private
    parameter layout. A later change to what a block computes — a different
    position encoding, a normalized query, another feed-forward — retires this
    helper and the test above it rather than updating either. Its failure means
    the blocks stopped being the wrapper, which after that point is the
    intent rather than a defect.
    """

    layer = nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=config.attention_heads,
        dim_feedforward=config.feedforward_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    wrapper = nn.TransformerEncoder(
        layer,
        num_layers=config.transformer_layers,
        norm=nn.LayerNorm(config.model_dim),
        enable_nested_tensor=False,
    )
    state: dict[str, torch.Tensor] = {
        "norm.weight": model.transformer_norm.weight,
        "norm.bias": model.transformer_norm.bias,
    }
    for index, module in enumerate(model.transformer_blocks):
        block = cast(TransformerBlock, module)
        feedforward_in = cast(nn.Linear, block.feedforward[0])
        feedforward_out = cast(nn.Linear, block.feedforward[3])
        state |= {
            f"layers.{index}.norm1.weight": block.attention_norm.weight,
            f"layers.{index}.norm1.bias": block.attention_norm.bias,
            f"layers.{index}.self_attn.in_proj_weight": (
                block.attention.qkv_projection.weight
            ),
            f"layers.{index}.self_attn.in_proj_bias": (
                block.attention.qkv_projection.bias
            ),
            f"layers.{index}.self_attn.out_proj.weight": (
                block.attention.output_projection.weight
            ),
            f"layers.{index}.self_attn.out_proj.bias": (
                block.attention.output_projection.bias
            ),
            f"layers.{index}.norm2.weight": block.feedforward_norm.weight,
            f"layers.{index}.norm2.bias": block.feedforward_norm.bias,
            f"layers.{index}.linear1.weight": feedforward_in.weight,
            f"layers.{index}.linear1.bias": feedforward_in.bias,
            f"layers.{index}.linear2.weight": feedforward_out.weight,
            f"layers.{index}.linear2.bias": feedforward_out.bias,
        }
    wrapper.load_state_dict(state)
    return wrapper


def _tiny_config(
    maximum_context_plies: int = 8,
    *,
    transformer_layers: int = 1,
    dropout: float = 0.0,
) -> MoveModelConfig:
    return MoveModelConfig(
        piece_embedding_dim=2,
        action_embedding_dim=4,
        model_dim=16,
        attention_heads=2,
        transformer_layers=transformer_layers,
        feedforward_dim=24,
        dropout=dropout,
        maximum_context_plies=maximum_context_plies,
    )
