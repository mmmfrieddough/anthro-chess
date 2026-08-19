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
    BOARD_SQUARE_COUNT,
    DecisionHistory,
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    collate_sequences,
    en_passant_token,
    encode_game,
    encoding_identity,
)
from anthro_chess.models import (
    MoveModel,
    MoveModelBatch,
    MoveModelConfig,
    OptionalTensor,
    RatingEmbedding,
    SourceDestinationHead,
)
from anthro_chess.models.move_model import ResidualBlock
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
    assert batch.inputs.repetition_count[0].tolist() == [0, 0, 0]
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
    model = MoveModel(_tiny_config())

    logits = model(batch)

    assert logits.shape == (2, 3, ACTION_VOCABULARY_SIZE)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    assert model.identity()["version"] == 7
    assert model.identity()["action_vocabulary"] == action_vocabulary_identity()
    assert model.identity()["encoding"] == encoding_identity()
    assert model.identity()["rating_conditioning"] == "square-token-input-embedding"
    assert model.identity()["timing_inputs"] is False
    assert model.identity()["timing_head"] is False


def test_padding_cannot_change_what_a_real_timestep_predicts() -> None:
    """What makes a padded logit ignorable rather than merely unread.

    A shorter history batched beside a longer one must see exactly what it
    would have seen alone.
    """

    torch.manual_seed(19)
    alone = MoveModelBatch.from_sequence_batch(_sequence_batch(("d2d4",)))
    padded = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("d2d4",), ("e2e4", "e7e5", "g1f3"))
    )
    model = MoveModel(_tiny_config()).eval()

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


@pytest.mark.parametrize(
    "position",
    [
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 5 4",
        # An en-passant square and one side's castling right already gone, so
        # the two whole-position columns the flip has to reorder are both
        # carrying something rather than sitting at their absent value.
        "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQq e3 0 3",
    ],
    ids=["open-game", "en-passant-and-lost-castling"],
)
def test_a_decision_and_its_mirror_image_score_the_same_chess(position: str) -> None:
    """The flip is only sound if undoing it lands on the same move.

    Every board is presented from the side to move, so a position with black to
    move and the same position with the colours swapped and the ranks mirrored
    reach the layers as one input. Their logits therefore have to agree under
    the same mirror on the action vocabulary, or the model is playing one
    position and answering about the other.

    The colour input is the one thing that genuinely differs between them, so it
    is held inert here; the test above covers that it reaches the model at all.
    """

    torch.manual_seed(67)
    model = MoveModel(_tiny_config(layers=2)).eval()
    with torch.no_grad():
        # A random weight everywhere, so no table can pass by accident: an
        # untrained model is symmetric enough that a wrong permutation could.
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3)
        model.square_encoder.side_embedding.weight.zero_()

    mirrored = chess.Board(position).mirror().fen()

    with torch.no_grad():
        black, white = (
            model(
                MoveModelBatch.from_decision_context(
                    DecisionHistory(initial_fen=fen).context(target_rating=1500)
                )
            )[0, -1]
            for fen in (position, mirrored)
        )

    mirror = torch.tensor(
        [
            encode_move(
                chess.Move(
                    move.from_square ^ 56,
                    move.to_square ^ 56,
                    promotion=move.promotion,
                )
            )
            for move in (decode_move(action) for action in range(MOVE_ACTION_COUNT))
        ]
    )
    torch.testing.assert_close(
        black[:MOVE_ACTION_COUNT][mirror],
        white[:MOVE_ACTION_COUNT],
        rtol=0.0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        black[MOVE_ACTION_COUNT:], white[MOVE_ACTION_COUNT:], rtol=0.0, atol=1e-5
    )


def test_the_side_playing_reaches_the_model_even_though_the_board_is_flipped() -> None:
    """Flipping erases which player is which, and human play is not symmetric.

    A repertoire as white is not the mirror of a repertoire as black, so the one
    bit the flip destroys is put back. Leela-CF carries it for the same reason.
    """

    torch.manual_seed(71)
    model = MoveModel(_tiny_config()).eval()
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(("e2e4", "e7e5")))
    swapped = MoveModelBatch.from_sequence_batch(_sequence_batch(("e2e4", "e7e5")))
    swapped.inputs.side_to_move[0, 0] = 1

    with torch.no_grad():
        assert not torch.equal(model(batch)[0, 0], model(swapped)[0, 0])


@pytest.mark.parametrize(
    ("history_positions", "reaches"),
    [(2, False), (4, True)],
    ids=["outside-the-stack", "inside-the-stack"],
)
def test_a_decision_reads_exactly_the_boards_its_stack_declares(
    history_positions: int,
    reaches: bool,
) -> None:
    """History is the token depth now, so what it reaches is what it holds."""

    torch.manual_seed(53)
    model = MoveModel(_tiny_config(history_positions=history_positions)).eval()
    moves = ("e2e4", "e7e5", "g1f3", "b8c6")
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    disturbed = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    disturbed.inputs.piece_ids[0, 0].zero_()

    with torch.no_grad():
        unchanged = torch.equal(model(batch)[0, 3], model(disturbed)[0, 3])

    assert unchanged is not reaches


def test_a_history_shorter_than_the_stack_repeats_its_earliest_board() -> None:
    """Every game opens with fewer boards behind it than the stack holds.

    Reaching before a row's first column clamps onto it, so a decision three
    plies into a game reads what a row of three copies of that same board would
    have given it.
    """

    torch.manual_seed(59)
    model = MoveModel(_tiny_config(history_positions=3)).eval()
    moves = ("e2e4", "e7e5", "g1f3")
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    repeated = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    inputs = repeated.inputs
    for column in (
        inputs.piece_ids,
        inputs.side_to_move,
        inputs.castling_rights,
        inputs.en_passant_token,
        inputs.halfmove_clock,
        inputs.fullmove_number,
        inputs.repetition_count,
        inputs.target_rating.values,
        inputs.target_rating.present,
    ):
        column[0, 1:] = column[0, :1]

    with torch.no_grad():
        opening = model.encode(batch, torch.tensor([[0]]))
        stacked = model.encode(repeated, torch.tensor([[2]]))

    torch.testing.assert_close(opening, stacked, rtol=0.0, atol=1e-6)


def test_a_repetition_the_rules_would_let_a_player_claim_reaches_the_model() -> None:
    """A model that cannot see a repetition cannot decide when to claim a draw."""

    torch.manual_seed(73)
    model = MoveModel(_tiny_config()).eval()
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(("e2e4", "e7e5")))
    repeated = MoveModelBatch.from_sequence_batch(_sequence_batch(("e2e4", "e7e5")))
    repeated.inputs.repetition_count[0, 1] = 2

    with torch.no_grad():
        assert not torch.equal(model(batch)[0, 1], model(repeated)[0, 1])


def test_history_is_truncated_only_while_training() -> None:
    """Opening plies are short, and training on nothing else would leave them odd.

    Every configuration would otherwise exercise this only in a real run, so a
    truncation wired to nothing would be invisible until then. Serving must not
    truncate at all.
    """

    torch.manual_seed(61)
    model = MoveModel(_tiny_config(history_positions=4, history_dropout=1.0))
    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3", "b8c6"))
    )

    with torch.no_grad():
        trained = [model.train()(batch) for _ in range(2)]
        evaluated = [model.eval()(batch) for _ in range(2)]

    assert not torch.equal(trained[0], trained[1])
    torch.testing.assert_close(evaluated[0], evaluated[1], rtol=0.0, atol=0.0)


def test_every_layer_is_drawn_on_its_own_rather_than_copied() -> None:
    """A stack built by cloning one prototype layer starts degenerate.

    Every block is drawn on its own, and a later collapse back to a copied
    prototype -- which is what the framework's own stack builder does -- would
    leave a multi-layer model beginning training as identical layers, with
    nothing else in the suite noticing.
    """

    torch.manual_seed(31)
    model = MoveModel(_tiny_config(layers=2))

    first, second = (
        cast(ResidualBlock, block).attention.qkv_projection.weight
        for block in model.blocks
    )

    assert not torch.equal(first, second)


def test_one_template_bank_serves_every_layer() -> None:
    """A template is 4096 values whatever the model width is.

    Held per layer instead, the bank would multiply by depth and dominate the
    parameter count outright, which is what makes a fixed depth affordable at
    all. Chessformer shares one bank across its whole model and so does this.
    """

    model = MoveModel(_tiny_config(layers=3))

    banks = [name for name, _ in model.named_parameters() if "template" in name]
    assert banks == ["bias_templates"]


def test_the_dropout_setting_reaches_attention_and_the_block_around_it() -> None:
    """Four framework-owned dropout sites became four written-out ones.

    Every configuration sets this to zero, so a site wired to nothing would be
    invisible until the first run that turns the dial up. Attention is checked
    on its own because its dropout is the one inside the fused attention call
    rather than a module the block composes.
    """

    torch.manual_seed(37)
    config = _tiny_config(dropout=0.5)
    model = MoveModel(config)
    templates = model.bias_templates
    block = cast(ResidualBlock, model.blocks[0])
    hidden = torch.randn(2, BOARD_SQUARE_COUNT, config.model_dim)

    with torch.no_grad():
        attention = [block.attention.train()(hidden, templates) for _ in range(2)]
        trained = [block.train()(hidden, templates) for _ in range(2)]
        evaluated = [block.eval()(hidden, templates) for _ in range(2)]

    assert not torch.equal(attention[0], attention[1])
    assert not torch.equal(trained[0], trained[1])
    torch.testing.assert_close(evaluated[0], evaluated[1], rtol=0.0, atol=0.0)


def test_the_forward_pass_compiles_whole_and_once_across_a_run_of_widths() -> None:
    """Length buckets vary the padded width, and none of them is a recompile.

    ``fullgraph`` is what makes a graph break an error here rather than a
    silently slower run.
    """

    torch._dynamo.reset()
    counter = CompileCounter()
    model = MoveModel(_tiny_config()).eval()
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
    model = MoveModel(_tiny_config()).to(backend).eval()
    batch = _batch_of_width(12, device=backend)
    compiled = torch.compile(model, fullgraph=True, mode=mode)

    with torch.no_grad():
        expected = model(batch)
        # Twice: a captured graph is replayed on the second call, not the first.
        compiled(batch)
        actual = compiled(batch)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_the_identity_carries_every_value_needed_to_rebuild_the_model() -> None:
    """A checkpoint is rebuilt from its identity, so a gap becomes a default."""

    config = _tiny_config()
    config_record = MoveModel(config).identity()["config"]

    assert config_record == config.model_dump(mode="json")
    assert MoveModelConfig.model_validate(config_record) == config
    assert (
        MoveModel(_tiny_config(history_positions=4)).identity()
        != MoveModel(config).identity()
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
    model = MoveModel(_tiny_config()).eval()

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


def test_rating_reaches_the_squares_rather_than_only_the_logits() -> None:
    """The fault `#177` measured, stated as the invariant that replaces it.

    The board representation itself moves with the rating, so every layer
    computes with it. An implementation that conditioned at the end --
    correcting a rating-neutral feature on the way out -- would still pass every
    shape and vocabulary test above, and would fail here.
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
    model = MoveModel(_tiny_config()).eval()
    decisions = torch.arange(3).unsqueeze(0)

    with torch.no_grad():
        unrated_squares = model.encode(unrated, decisions)
        rated_squares = model.encode(rated, decisions)
        unrated_logits = model(unrated)
        rated_logits = model(rated)

    assert not torch.equal(unrated_squares, rated_squares)
    assert not torch.equal(unrated_logits, rated_logits)


def test_reading_one_decision_agrees_with_reading_the_whole_pass() -> None:
    """The serving shortcut is sound only while it is the same arithmetic.

    Serving names one ply per row and runs the model on that decision alone,
    which is valid because every stage reads one decision at a time. A later
    change letting any of them reach across decisions would make the two
    disagree, and every served move would quietly stop matching the model that
    was trained.
    """

    torch.manual_seed(41)
    model = MoveModel(_tiny_config()).eval()
    batch = MoveModelBatch.from_sequence_batch(
        _sequence_batch(("e2e4", "e7e5", "g1f3"), ("d2d4",), white_rating=1500)
    )
    decisions = torch.tensor([2, 0])

    with torch.no_grad():
        whole = model(batch)
        expected = whole[torch.arange(decisions.shape[0]), decisions]
        actual = model.decide_at(batch, decisions)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


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
    nothing else in the suite notices.

    So the board is rebuilt here by two-dimensional indexing while the head
    reaches it through a packed flat offset. A transposed pair or a wrong stride
    passes any check that repeats the packing arithmetic and fails this one.
    """

    torch.manual_seed(43)
    config = _tiny_config()
    head = SourceDestinationHead(config)
    squares = torch.randn(1, 1, BOARD_SQUARE_COUNT, config.model_dim)
    white = torch.zeros((1, 1), dtype=torch.bool)

    with torch.no_grad():
        logits = head(squares, white)[0, 0]
        sources = head.source_projection(squares)[0, 0]
        destinations = head.destination_projection(squares)[0, 0]
        board = sources @ destinations.transpose(-1, -2) * config.model_dim**-0.5
        promotions = head.promotion_projection(squares)[0, 0]

    for action_id in range(MOVE_ACTION_COUNT):
        move = decode_move(action_id)
        expected = board[move.from_square, move.to_square]
        if move.promotion is not None:
            expected = (
                expected + promotions[move.to_square, move.promotion - chess.KNIGHT]
            )
        assert logits[action_id] == pytest.approx(expected.item(), abs=1e-5)


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
    squares = torch.randn(1, 1, BOARD_SQUARE_COUNT, config.model_dim)
    white = torch.zeros((1, 1), dtype=torch.bool)
    queen = encode_move(chess.Move.from_uci("a7a8q"))
    knight = encode_move(chess.Move.from_uci("a7a8n"))
    # The same square pair as the promotions above, so only the zero-padded
    # promotion column separates it from them. A quiet arrival at a different
    # square would be kept clean by its destination rather than by the pad.
    quiet = encode_move(chess.Move.from_uci("a7a8"))

    with torch.no_grad():
        before = head(squares, white)
        head.promotion_projection.bias[chess.QUEEN - chess.KNIGHT] += 5.0
        after = head(squares, white)

    assert after[0, 0, queen] - before[0, 0, queen] == pytest.approx(5.0, abs=1e-4)
    assert after[0, 0, knight] == pytest.approx(before[0, 0, knight].item(), abs=1e-5)
    assert after[0, 0, quiet] == pytest.approx(before[0, 0, quiet].item(), abs=1e-5)


def test_terminal_actions_are_scored_from_the_whole_board_not_a_square_pair() -> None:
    """Resignation and a draw claim carry no move, so they read no square pair."""

    torch.manual_seed(23)
    config = _tiny_config()
    head = SourceDestinationHead(config)
    squares = torch.randn(1, 1, BOARD_SQUARE_COUNT, config.model_dim)
    white = torch.zeros((1, 1), dtype=torch.bool)

    with torch.no_grad():
        logits = head(squares, white)
        terminal = head.terminal_projection(squares.mean(dim=-2))

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
    model = MoveModel(config)
    attention = cast(ResidualBlock, model.blocks[0]).attention
    tokens = torch.randn(3, BOARD_SQUARE_COUNT, config.model_dim)

    with torch.no_grad():
        generated = attention.geometric_bias(tokens, model.bias_templates)

    assert generated.shape == (
        3,
        config.attention_heads,
        BOARD_SQUARE_COUNT,
        BOARD_SQUARE_COUNT,
    )
    assert torch.count_nonzero(generated) == 0


@pytest.mark.parametrize(
    "moves",
    [
        ("e2e4",),
        ("e2e4", "e7e5", "g1f3"),
    ],
    ids=["single-decision", "short-game"],
)
def test_ordinary_model_and_loss_path_overfits_a_fixed_tiny_sample(
    moves: tuple[str, ...],
) -> None:
    torch.manual_seed(11)
    batch = MoveModelBatch.from_sequence_batch(_sequence_batch(moves))
    assert not torch.any(batch.inputs.target_rating.present)
    model = MoveModel(_tiny_config())
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
    *,
    layers: int = 1,
    dropout: float = 0.0,
    history_positions: int = 3,
    history_dropout: float = 0.0,
) -> MoveModelConfig:
    """Return a model small enough to train in a test, and deterministic.

    ``history_dropout`` is off by default so that a test comparing two forward
    passes is comparing the model rather than two draws of the training-time
    history truncation, which has a test of its own below.
    """

    return MoveModelConfig(
        piece_embedding_dim=2,
        model_dim=16,
        attention_heads=2,
        layers=layers,
        feedforward_dim=24,
        history_positions=history_positions,
        history_dropout=history_dropout,
        geometric_token_dim=4,
        geometric_bias_dim=8,
        dropout=dropout,
    )
