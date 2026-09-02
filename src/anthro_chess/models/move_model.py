"""Action model over one decision's board and the boards stacked behind it.

One forward pass is one decision. A position is read as 64 square tokens whose
depth carries the last few boards, flipped so the side to move is always the
one playing up the board, and a stack of spatial layers attends between those
tokens along a geometric bias generated from the position itself. Decisions
0066 and 0070 record why each of those is what it is.
"""

from __future__ import annotations

from functools import cache
from typing import cast

import chess
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anthro_chess.chess import (
    MOVE_ACTION_COUNT,
    TERMINAL_ACTION_IDS,
    action_vocabulary_identity,
    decode_move,
)
from anthro_chess.data import (
    BOARD_SQUARE_COUNT,
    EN_PASSANT_TOKEN_COUNT,
    REPETITION_STATE_COUNT,
    en_passant_token,
    encoding_identity,
)
from anthro_chess.models.batching import MoveModelBatch, OptionalTensor
from anthro_chess.models.config import MoveModelConfig

_PIECE_ID_COUNT = 13
_SIDE_TO_MOVE_COUNT = 2
_CASTLING_RIGHTS_COUNT = 16
_PROMOTION_CHOICE_COUNT = 4
#: The ratings the two anchor embeddings stand for. Every rating is placed on
#: the segment between them, so these bound where the dial can move: at or below
#: the weak anchor and at or above the strong one, turning it further does
#: nothing.
#:
#: Set wide. The span is close to a reparameterization -- the distance between
#: two ratings' embeddings is their difference in interpolation weight times the
#: learned gap between the anchors, and the second factor absorbs the first, so
#: narrowing this buys far less resolution than it appears to. Clamping is the
#: one effect training cannot absorb, because it makes a stretch of the dial
#: inert. Rating scales also differ -- Lichess blitz is not FIDE and not Lichess
#: bullet -- so the range that must not clamp is wider than any one corpus.
#: The corpus reaches into the 3900s, so both ends sit outside any rating it
#: carries.
_WEAK_RATING_ANCHOR = 0.0
_STRONG_RATING_ANCHOR = 5000.0


class RatingEmbedding(nn.Module):
    """Place a rating on the segment between a weak and a strong anchor.

    Two learned endpoints and a linear interpolation between them, rather than a
    network free to map ratings wherever it likes. The dial's whole job is to be
    ordered, and an unconstrained map is under no obligation to keep it that way:
    `#177` measured 900 configured Elo arriving as a 12-Elo span with the
    ordering itself no better than chance. Interpolation makes the representation
    monotone in the rating by construction, which is the one property the product
    needs and the one the free map never had to learn.
    """

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.weak = nn.Parameter(torch.empty(config.model_dim))
        self.strong = nn.Parameter(torch.empty(config.model_dim))
        self.unrated = nn.Parameter(torch.empty(config.model_dim))
        for anchor in (self.weak, self.strong, self.unrated):
            nn.init.normal_(anchor, std=0.02)

    def forward(self, rating: OptionalTensor) -> Tensor:
        """Return one embedding per decision, shaped batch by decision by width."""

        span = _STRONG_RATING_ANCHOR - _WEAK_RATING_ANCHOR
        weakness = (
            ((_STRONG_RATING_ANCHOR - rating.values.float()) / span)
            .clamp(0.0, 1.0)
            .unsqueeze(-1)
        )
        placed = weakness * self.weak + (1.0 - weakness) * self.strong
        return torch.where(rating.present.unsqueeze(-1), placed, self.unrated)


class SquareTokenEncoder(nn.Module):
    """Build the 64 square tokens one decision is read as.

    Every token carries its own square across the stacked boards, the rule state
    that applies to the whole position, and the rating, so the layers above this
    see a board whose geometry survived, the moves that produced it, and a
    decision-maker whose strength is already part of the representation.

    Each board is oriented to whoever was to move when it was current, so the
    slots alternate between the two players' frames rather than being rotated
    into the decision's. A slot reaching before the row repeats a board and
    breaks that alternation, which is what a game's opening plies present too.
    """

    mirrored_squares: Tensor
    flipped_piece_ids: Tensor
    flipped_castling_rights: Tensor
    flipped_en_passant_tokens: Tensor
    repetition_thresholds: Tensor

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        embedding_dim = config.piece_embedding_dim
        self.history = config.history_positions
        self.history_dropout = config.history_dropout
        self.piece_embedding = nn.Embedding(_PIECE_ID_COUNT, embedding_dim)
        self.side_embedding = nn.Embedding(_SIDE_TO_MOVE_COUNT, embedding_dim)
        self.castling_embedding = nn.Embedding(_CASTLING_RIGHTS_COUNT, embedding_dim)
        self.en_passant_embedding = nn.Embedding(EN_PASSANT_TOKEN_COUNT, embedding_dim)
        self.square_identity = nn.Parameter(
            torch.empty(BOARD_SQUARE_COUNT, config.model_dim)
        )
        nn.init.normal_(self.square_identity, std=0.02)
        position_dim = (
            3 * embedding_dim + 2 + self.history * (REPETITION_STATE_COUNT - 1)
        )
        # Two projections summed rather than one over a concatenation. A linear
        # map over concatenated inputs is exactly the sum of its parts, and the
        # position half is the same value on all 64 squares -- projecting it
        # once and broadcasting the result costs a fraction of replicating it
        # across the board first and is the same function.
        self.piece_projection = nn.Linear(
            self.history * embedding_dim,
            config.model_dim,
            bias=False,
        )
        self.position_projection = nn.Linear(position_dim, config.model_dim)
        self.normalization = nn.LayerNorm(config.model_dim)
        self.rating_embedding = RatingEmbedding(config)
        # Derived constants, not state: pure functions of the encoding, so they
        # do not belong in a checkpoint.
        for name, values in (
            ("mirrored_squares", chess.SQUARES_180),
            ("flipped_piece_ids", _flipped_piece_ids()),
            ("flipped_castling_rights", _flipped_castling_rights()),
            ("flipped_en_passant_tokens", _flipped_en_passant_tokens()),
            ("repetition_thresholds", tuple(range(1, REPETITION_STATE_COUNT))),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.long),
                persistent=False,
            )

    def forward(self, batch: MoveModelBatch, decisions: Tensor) -> Tensor:
        """Return tokens shaped batch by decision by square by width."""

        inputs = batch.inputs
        black = inputs.side_to_move.bool()
        boards = torch.where(
            black.unsqueeze(-1),
            self.flipped_piece_ids[inputs.piece_ids[..., self.mirrored_squares]],
            inputs.piece_ids,
        )
        history = self._history_index(
            decisions, _at_decision(batch.history_floor, decisions)
        )
        # Squares before slots, so the embedding that follows leaves the stack
        # in the last dimension and the projection reads it as one contiguous
        # depth rather than transposing a five-dimensional tensor to get there.
        stacked = _gather_plies(boards, history).transpose(-1, -2)
        repetitions = _gather_plies(inputs.repetition_count, history)
        tokens = self.piece_projection(self.piece_embedding(stacked).flatten(-2))

        decided_side = _at_decision(inputs.side_to_move, decisions)
        decided_black = decided_side.bool()
        # A transform rather than a token vocabulary, so it stays here while the
        # two nullable inputs moved. Decision 0035 says why.
        rule_counts = torch.log1p(
            torch.stack(
                (
                    _at_decision(inputs.halfmove_clock, decisions).float(),
                    _at_decision(inputs.fullmove_number, decisions).float(),
                ),
                dim=-1,
            )
        )
        position = torch.cat(
            (
                self.side_embedding(decided_side),
                self.castling_embedding(
                    self._oriented(
                        _at_decision(inputs.castling_rights, decisions),
                        decided_black,
                        self.flipped_castling_rights,
                    )
                ),
                self.en_passant_embedding(
                    self._oriented(
                        _at_decision(inputs.en_passant_token, decisions),
                        decided_black,
                        self.flipped_en_passant_tokens,
                    )
                ),
                rule_counts,
                (repetitions.unsqueeze(-1) >= self.repetition_thresholds)
                .to(rule_counts.dtype)
                .flatten(-2),
            ),
            dim=-1,
        )
        tokens = tokens + self.position_projection(position).unsqueeze(-2)
        tokens = tokens + self.square_identity
        rating = _at_decision_rating(inputs.target_rating, decisions)
        tokens = tokens + self.rating_embedding(rating).unsqueeze(-2)
        return cast(Tensor, self.normalization(tokens))

    def _history_index(self, decisions: Tensor, floor: Tensor) -> Tensor:
        """Return the ply each history slot of each decision reads.

        Reaching before the decision's own game is not an error: the earliest
        board available is repeated, which is what a game's opening plies
        present anyway and what Chessformer trains against. ``floor`` is what
        stops a row holding several games letting one decision read the game in
        front of it.
        """

        offsets = torch.arange(self.history, device=decisions.device)
        reach = offsets.expand(*decisions.shape, -1)
        if self.training and self.history_dropout > 0.0:
            kept = torch.randint(
                self.history,
                decisions.shape,
                device=decisions.device,
            )
            truncated = (
                torch.rand(decisions.shape, device=decisions.device)
                < self.history_dropout
            )
            limit = torch.where(truncated, kept, self.history - 1)
            reach = torch.minimum(reach, limit.unsqueeze(-1))
        return torch.maximum(decisions.unsqueeze(-1) - reach, floor.unsqueeze(-1))

    def _oriented(self, values: Tensor, black: Tensor, flipped: Tensor) -> Tensor:
        """Return one whole-position column read from the mover's own side."""

        return torch.where(black, flipped[values], values)


class GeometricAttentionBias(nn.Module):
    """Generate a per-head square-by-square attention bias from the position.

    Dot-product attention between square tokens has no idea which squares are
    adjacent, on a diagonal, or a knight's move apart, and a fixed positional
    encoding cannot say which of those relations matters in *this* position — a
    pinned bishop's diagonal is load-bearing and an empty one is not. This reads
    the whole board down to a small vector and mixes a bank of 64-by-64
    templates from it, so the geometry each head attends along is chosen per
    position.

    The bank of templates is the model's and is passed in, so every layer mixes
    one shared vocabulary of square relations and only the mixing is its own.
    """

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        bias_dim = config.geometric_bias_dim
        self.compression = nn.Linear(config.model_dim, config.geometric_token_dim)
        self.board = nn.Sequential(
            nn.Linear(BOARD_SQUARE_COUNT * config.geometric_token_dim, bias_dim),
            nn.GELU(),
            nn.LayerNorm(bias_dim),
            nn.Linear(bias_dim, self.heads * bias_dim),
            nn.GELU(),
        )
        self.mixture_norm = nn.LayerNorm(bias_dim)

    def forward(self, hidden: Tensor, templates: Tensor) -> Tensor:
        """Return biases shaped position by head by square by square."""

        compressed = self.compression(hidden).flatten(-2)
        mixture = self.mixture_norm(
            self.board(compressed).unflatten(-1, (self.heads, -1))
        )
        return cast(
            Tensor,
            (mixture @ templates.transpose(-1, -2)).unflatten(
                -1,
                (BOARD_SQUARE_COUNT, BOARD_SQUARE_COUNT),
            ),
        )


class SpatialAttention(nn.Module):
    """Attend between the squares of one position, along learned geometry."""

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.model_dim // config.attention_heads
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.model_dim)
        self.geometric_bias = GeometricAttentionBias(config)
        # ``nn.Linear``'s own default scales from fan-in alone, which reads this
        # fused weight as one projection rather than the three it fans out into.
        nn.init.xavier_uniform_(self.qkv_projection.weight)
        nn.init.zeros_(self.qkv_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, hidden: Tensor, templates: Tensor) -> Tensor:
        """Return attended features, batch dimension first."""

        query, key, value = (
            self.qkv_projection(hidden)
            .unflatten(-1, (3, self.heads, self.head_dim))
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            attn_mask=self.geometric_bias(hidden, templates).to(query.dtype),
        )
        return cast(
            Tensor,
            self.output_projection(attended.transpose(1, 2).flatten(-2)),
        )


class ResidualBlock(nn.Module):
    """One pre-norm residual pair: spatial attention, then a feed-forward."""

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = SpatialAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim),
        )
        # Dropout carries no state, so both residuals read the same module.
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, templates: Tensor) -> Tensor:
        """Return the block's output for a whole batch of positions."""

        hidden = hidden + self.residual_dropout(
            self.attention(self.attention_norm(hidden), templates)
        )
        hidden = hidden + self.residual_dropout(
            self.feedforward(self.feedforward_norm(hidden))
        )
        return hidden


class SourceDestinationHead(nn.Module):
    """Score a move as the attention from its source square to its destination.

    The vocabulary this project speaks to the rest of the system is unchanged: a
    flat action id, because legal masking, UCI, and every benchmark are written
    in it. What changes is where a move's logit comes from. A source-square query
    against a destination-square key means the head is expressed in the same
    board geometry the encoder produced, and a move it has never seen still
    scores from squares it has, which a flat projection over a fixed vocabulary
    cannot do.

    The squares arrive in the mover's frame and the vocabulary is written in the
    board's, so this is also where the flip is undone: reordering the tokens is
    the same map as reordering the board it would produce, and it is 64 values
    per position rather than 4096.
    """

    mirrored_squares: Tensor
    move_square_slots: Tensor
    move_promotion_slots: Tensor

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.scale = config.model_dim**-0.5
        self.source_projection = nn.Linear(config.model_dim, config.model_dim)
        self.destination_projection = nn.Linear(config.model_dim, config.model_dim)
        self.promotion_projection = nn.Linear(
            config.model_dim,
            _PROMOTION_CHOICE_COUNT,
        )
        self.terminal_projection = nn.Linear(
            config.model_dim,
            len(TERMINAL_ACTION_IDS),
        )
        square_slots, promotion_slots = _move_index_tables()
        # Derived constants, not state: pure functions of the action
        # vocabulary, so they do not belong in a checkpoint.
        for name, values in (
            ("mirrored_squares", chess.SQUARES_180),
            ("move_square_slots", square_slots),
            ("move_promotion_slots", promotion_slots),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.long),
                persistent=False,
            )

    def forward(self, squares: Tensor, black_to_move: Tensor) -> Tensor:
        """Return logits over the whole action vocabulary, moves then terminals."""

        squares = torch.where(
            black_to_move.unsqueeze(-1).unsqueeze(-1),
            squares.index_select(-2, self.mirrored_squares),
            squares,
        )
        sources = self.source_projection(squares)
        destinations = self.destination_projection(squares)
        board = (sources @ destinations.transpose(-1, -2)) * self.scale
        # The padded column is a constant zero rather than a parameter, so a
        # move that promotes nothing reads an exact zero instead of a weight
        # that has to learn to be one.
        promotions = F.pad(self.promotion_projection(squares), (0, 1))
        moves = (
            board.flatten(-2)[..., self.move_square_slots]
            + promotions.flatten(-2)[..., self.move_promotion_slots]
        )
        terminals = self.terminal_projection(squares.mean(dim=-2))
        return torch.cat((moves, terminals), dim=-1)


class MoveModel(nn.Module):
    """Predict human action logits from one decision and the boards behind it."""

    def __init__(self, config: MoveModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoveModelConfig()
        self.square_encoder = SquareTokenEncoder(self.config)
        # One bank for the whole model, handed down rather than held by each
        # layer, so a checkpoint carries it once. It starts at zero, which makes
        # a fresh model ordinary dot-product attention: the geometry is
        # something training adds rather than something it has to first undo.
        self.bias_templates = nn.Parameter(
            torch.zeros(
                BOARD_SQUARE_COUNT * BOARD_SQUARE_COUNT,
                self.config.geometric_bias_dim,
            )
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(self.config) for _ in range(self.config.layers)
        )
        # A pre-norm block leaves its output unnormalized for whatever follows,
        # so the stack ends with one. Named rather than the last element of a
        # sequence, because a checkpoint key that moves with the layer count
        # cannot be read without also reading the configuration.
        self.norm = nn.LayerNorm(self.config.model_dim)
        self.action_head = SourceDestinationHead(self.config)

    def forward(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits shaped batch by sequence by vocabulary.

        Every batch is validated by whichever factory built it, so nothing is
        rechecked here. Padded timesteps produce ordinary finite logits that no
        caller reads: the loss ignores them by target, and every scoring path
        gathers by the loss mask before looking.
        """

        return self.decide(batch, batch.decision_columns)

    def decide_at(self, batch: MoveModelBatch, decisions: Tensor) -> Tensor:
        """Return logits for one named ply per row, shaped batch by vocabulary.

        Serving reads a single decision out of each history, and every stage of
        this model is per-decision, so naming the ply costs one position per row
        rather than one per ply of the game so far.
        """

        return self.decide(batch, decisions.unsqueeze(-1))[:, 0]

    def decide(self, batch: MoveModelBatch, decisions: Tensor) -> Tensor:
        """Return logits for the named decisions, shaped like their index."""

        squares = self.encode(batch, decisions)
        black = _at_decision(batch.inputs.side_to_move, decisions).bool()
        return cast(Tensor, self.action_head(squares, black))

    def encode(self, batch: MoveModelBatch, decisions: Tensor) -> Tensor:
        """Encode the named decisions' squares, each reading its own history."""

        tokens = self.square_encoder(batch, decisions)
        positions = tokens.flatten(0, 1)
        for block in self.blocks:
            positions = block(positions, self.bias_templates)
        return cast(
            Tensor,
            self.norm(positions).unflatten(0, tokens.shape[:2]),
        )

    def identity(self) -> dict[str, object]:
        """Return compatibility metadata for future runs and checkpoints."""

        return model_identity(self.config)


def parameter_count(config: MoveModelConfig) -> int:
    """Return how many trainable values a model built from ``config`` holds.

    Every tensor the assembled model owns, the action head included. The head
    is the part an outside count is most often missing, and here it runs from a
    sixty-third of the total at ``model_dim`` 32 to a thirty-ninth at 512: a
    small share that grows with width, because this head factors an action into
    two square choices rather than projecting onto the vocabulary.

    Built rather than derived from the shape, so it cannot drift away from what
    the architecture actually allocates.
    """

    return sum(parameter.numel() for parameter in MoveModel(config).parameters())


def model_identity(config: MoveModelConfig) -> dict[str, object]:
    """Return the compatibility metadata a model built from ``config`` carries.

    The whole configuration is carried, so a checkpoint says what to rebuild
    without a second record having to supply the rest. Anything left out here
    is a value the runner cannot recover, and it would then rebuild the model
    at that field's default instead of the run's.

    A function of the configuration rather than of a built model, because
    every field here is one: a caller comparing an artifact against what this
    code would produce should not have to allocate a network to do it.
    """

    return {
        "name": "anthro-move-model",
        # Version 7 is the architecture 0070 records: one decision per forward
        # pass, history stacked into the square tokens, and the board flipped to
        # the side to move. No version 6 checkpoint can be read.
        "version": 7,
        "config": config.model_dump(mode="json"),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "rating_conditioning": "square-token-input-embedding",
        # The anchors are not config, but they decide what every stored rating
        # means: move one and a retained checkpoint's dial silently reinterprets
        # itself. Carrying them here is what makes the runner's identity check
        # refuse such a checkpoint instead of serving it under new endpoints.
        "rating_anchors": [_WEAK_RATING_ANCHOR, _STRONG_RATING_ANCHOR],
        "timing_inputs": False,
        "timing_head": False,
    }


def _at_decision(values: Tensor, decisions: Tensor) -> Tensor:
    """Return one per-ply column read at each decision."""

    return values.gather(1, decisions)


def _at_decision_rating(rating: OptionalTensor, decisions: Tensor) -> OptionalTensor:
    """Return the nullable rating each decision's own mover carries."""

    return OptionalTensor(
        _at_decision(rating.values, decisions),
        _at_decision(rating.present, decisions),
    )


def _gather_plies(values: Tensor, history: Tensor) -> Tensor:
    """Return one per-ply column read at every history slot of every decision."""

    rows = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1)
    return values[rows, history]


@cache
def _flipped_piece_ids() -> tuple[int, ...]:
    """Return each piece id with its colour swapped, and empty left alone."""

    kinds = (_PIECE_ID_COUNT - 1) // 2
    return (
        0,
        *(
            piece + kinds if piece <= kinds else piece - kinds
            for piece in range(1, _PIECE_ID_COUNT)
        ),
    )


@cache
def _flipped_castling_rights() -> tuple[int, ...]:
    """Return each castling mask with the two players' rights exchanged."""

    return tuple(
        ((rights & 0b0011) << 2) | ((rights & 0b1100) >> 2)
        for rights in range(_CASTLING_RIGHTS_COUNT)
    )


@cache
def _flipped_en_passant_tokens() -> tuple[int, ...]:
    """Return each en-passant token with its square mirrored, absence aside."""

    return (
        en_passant_token(None),
        *(en_passant_token(square) for square in chess.SQUARES_180),
    )


@cache
def _move_index_tables() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return where each move id reads its board and promotion logit.

    The head produces a square-by-square board and a per-destination promotion
    bias; the vocabulary is a flat list. These are the gather that joins them,
    derived from the vocabulary itself so the two can never drift, and cached
    because every model built walks the whole vocabulary to produce them.
    """

    square_slots = []
    promotion_slots = []
    for action_id in range(MOVE_ACTION_COUNT):
        move = decode_move(action_id)
        square_slots.append(move.from_square * BOARD_SQUARE_COUNT + move.to_square)
        # A move that promotes nothing reads the zero column
        # ``SourceDestinationHead`` pads on, one past the four choices, so the
        # padded width rather than the choice count is the row stride.
        choice = (
            _PROMOTION_CHOICE_COUNT
            if move.promotion is None
            else move.promotion - chess.KNIGHT
        )
        promotion_slots.append(move.to_square * (_PROMOTION_CHOICE_COUNT + 1) + choice)
    return tuple(square_slots), tuple(promotion_slots)
