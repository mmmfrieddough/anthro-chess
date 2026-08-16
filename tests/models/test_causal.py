from __future__ import annotations

import math
from typing import cast

import chess
import pytest
import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounter

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    MOVE_ACTION_COUNT,
    RESIGNATION_ACTION_ID,
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
    en_passant_token,
    encode_game,
    encoding_identity,
    previous_action_token,
)
from anthro_chess.models import (
    CausalMoveModel,
    MoveModelBatch,
    MoveModelConfig,
    OptionalTensor,
    RatingEmbedding,
    SourceDestinationHead,
)
from anthro_chess.models.causal import (
    CausalSelfAttention,
    GeometricAttentionBias,
    ResidualBlock,
)
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
    assert batch.inputs.en_passant_token[0].tolist() == [
        en_passant_token(None),
        en_passant_token(chess.E3),
        en_passant_token(chess.E6),
    ]
    assert batch.inputs.halfmove_clock[0].tolist() == [0, 0, 0]
    assert batch.inputs.fullmove_number[0].tolist() == [1, 1, 2]
    assert batch.inputs.previous_action_token[0].tolist() == [
        previous_action_token(None),
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
    assert model.identity()["version"] == 6
    assert model.identity()["action_vocabulary"] == action_vocabulary_identity()
    assert model.identity()["encoding"] == encoding_identity()
    assert model.identity()["rating_conditioning"] == "square-token-input-embedding"
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
        cast(
            CausalSelfAttention, cast(ResidualBlock, block).attention
        ).qkv_projection.weight
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
    block = cast(ResidualBlock, CausalMoveModel(config).transformer_blocks[0])
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


def test_rating_reaches_the_trajectory_rather_than_only_the_decision() -> None:
    """The fault `#177` measured, stated as the invariant that replaces it.

    Version 5 encoded the history rating-neutrally and modulated the finished
    feature, so the trunk could not let rating change which history it attended
    to. This asserts the opposite: the encoded trajectory itself moves with the
    rating. An implementation that regressed to late conditioning would still
    pass every shape and vocabulary test above, and would fail here.
    """

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
        unrated_trajectory = model.encode_trajectory(unrated)
        rated_trajectory = model.encode_trajectory(rated)
        unrated_logits = model(unrated)
        rated_logits = model(rated)

    assert not torch.equal(unrated_trajectory, rated_trajectory)
    assert not torch.equal(unrated_logits, rated_logits)


def test_the_rating_embedding_moves_monotonically_along_one_axis() -> None:
    """What the interpolated anchors buy over a free map, stated directly.

    The dial has to be ordered to mean anything, and `#177` measured an ordering
    no better than chance. Every rating lands on the segment between the two
    anchors, so a higher rating is always further along it -- a property of the
    parameterization rather than something the loss has to discover.
    """

    torch.manual_seed(13)
    embedding = RatingEmbedding(_tiny_config())
    ratings = torch.tensor([[800, 1200, 1600, 2000, 2400]])

    with torch.no_grad():
        placed = embedding(
            OptionalTensor(ratings, torch.ones_like(ratings, dtype=torch.bool))
        )

    steps = placed[0, 1:] - placed[0, :-1]
    direction = steps[0] / steps[0].norm()
    projections = torch.stack([step @ direction for step in steps])
    assert torch.all(projections > 0)
    torch.testing.assert_close(
        steps,
        direction.unsqueeze(0) * projections.unsqueeze(-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_an_absent_rating_is_its_own_embedding_rather_than_a_rating_of_zero() -> None:
    """Missing has to stay explicit, or unrated play reads as maximally weak."""

    torch.manual_seed(17)
    embedding = RatingEmbedding(_tiny_config())
    values = torch.tensor([[0, 1500]])

    with torch.no_grad():
        absent = embedding(OptionalTensor(values, torch.tensor([[False, False]])))
        present = embedding(OptionalTensor(values, torch.tensor([[True, True]])))

    torch.testing.assert_close(absent[0, 0], absent[0, 1], rtol=0.0, atol=0.0)
    assert not torch.equal(absent[0, 0], present[0, 0])


def test_every_move_reads_the_head_square_pair_its_action_id_names() -> None:
    """The gather joining a square-by-square board to a flat vocabulary.

    The head scores a move as source-square against destination-square, while
    legal masking, UCI, and every benchmark speak flat action ids. If those two
    disagree by even one entry the model trains toward the wrong move and
    nothing else in the suite notices, so the mapping is checked against the
    vocabulary itself rather than against a second copy of the arithmetic.
    """

    head = SourceDestinationHead(_tiny_config())

    for action_id in range(MOVE_ACTION_COUNT):
        move = decode_move(action_id)
        assert int(head.move_square_slots[action_id]) == move.from_square * 64 + (
            move.to_square
        )
        choice = 4 if move.promotion is None else move.promotion - chess.KNIGHT
        assert int(head.move_promotion_slots[action_id]) == (
            move.to_square * 5 + choice
        )


def test_a_promotion_choice_moves_only_the_promotions_that_choose_it() -> None:
    """The promotion bias is per destination and per piece, not per move.

    Four promotions share one source and one destination, so the square pair
    alone cannot separate them. This is what does, and a bias leaking onto the
    quiet moves into the same square would make the head unable to tell a
    promotion from an ordinary arrival there.
    """

    torch.manual_seed(19)
    config = _tiny_config()
    head = SourceDestinationHead(config)
    squares = torch.randn(1, 1, 64, config.model_dim)
    hidden = torch.randn(1, 1, config.model_dim)
    queen = encode_move(chess.Move.from_uci("a7a8q"))
    knight = encode_move(chess.Move.from_uci("a7a8n"))
    quiet = encode_move(chess.Move.from_uci("a6a7"))

    with torch.no_grad():
        before = head(squares, hidden)
        head.promotion_projection.bias[chess.QUEEN - chess.KNIGHT] += 5.0
        after = head(squares, hidden)

    assert after[0, 0, queen] - before[0, 0, queen] == pytest.approx(5.0, abs=1e-4)
    assert after[0, 0, knight] == pytest.approx(before[0, 0, knight].item(), abs=1e-5)
    assert after[0, 0, quiet] == pytest.approx(before[0, 0, quiet].item(), abs=1e-5)


def test_terminal_actions_are_scored_from_the_history_not_from_a_square_pair() -> None:
    """Resignation and a draw claim carry no move, so they read no squares."""

    torch.manual_seed(23)
    config = _tiny_config()
    head = SourceDestinationHead(config)
    squares = torch.randn(1, 1, 64, config.model_dim)
    hidden = torch.randn(1, 1, config.model_dim)

    with torch.no_grad():
        logits = head(squares, hidden)
        terminal = head.terminal_projection(hidden)

    assert logits.shape[-1] == ACTION_VOCABULARY_SIZE
    torch.testing.assert_close(
        logits[..., RESIGNATION_ACTION_ID], terminal[..., 0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        logits[..., DRAW_CLAIM_ACTION_ID], terminal[..., 1], rtol=0.0, atol=0.0
    )


def test_the_geometric_bias_starts_as_no_bias_at_all() -> None:
    """A fresh model is ordinary dot-product attention over the squares.

    Training adds the geometry rather than first having to undo a random one,
    which is what the zeroed output layer buys. A default initialization here
    would perturb every attention logit of every spatial layer at step zero.
    """

    torch.manual_seed(29)
    config = _tiny_config()
    bias = GeometricAttentionBias(config)
    tokens = torch.randn(3, 64, config.model_dim)

    with torch.no_grad():
        generated = bias(tokens)

    assert generated.shape == (3, config.attention_heads, 64, 64)
    assert torch.count_nonzero(generated) == 0


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
        spatial_layers=1,
        transformer_layers=transformer_layers,
        decision_layers=1,
        feedforward_dim=24,
        geometric_token_dim=4,
        geometric_bias_dim=8,
        dropout=dropout,
        maximum_context_plies=maximum_context_plies,
    )
