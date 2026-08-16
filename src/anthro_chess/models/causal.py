"""Action-only causal model over exact per-ply chess context.

Three stages, in the order a decision passes through them: a spatial encoder
over the 64 square tokens of each position, a causal trunk over the ply axis,
and a spatial decoder that reads the trunk's history feature back onto the
squares the move head scores. Rating is embedded into the square tokens before
the first stage, so every layer computes with it rather than being corrected
afterwards. Decision 0066 records why each of those is what it is.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any, cast

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
    PREVIOUS_ACTION_TOKEN_COUNT,
    encoding_identity,
)
from anthro_chess.models.batching import MoveModelBatch, OptionalTensor
from anthro_chess.models.config import MoveModelConfig

_PIECE_ID_COUNT = 13
_SIDE_TO_MOVE_COUNT = 2
_CASTLING_RIGHTS_COUNT = 16
_PROMOTION_CHOICE_COUNT = 4
#: The ratings the two anchor embeddings stand for. Every rating is placed on
#: the segment between them, so these bound where the dial can move rather than
#: naming values the corpus is expected to contain: at or below the weak anchor
#: and at or above the strong one, turning the dial further does nothing.
#:
#: They bracket the human range rather than starting at zero. A weak anchor at
#: rating zero would spend most of the segment on ratings no player has, leaving
#: the range the corpus actually covers compressed into part of it — a
#: resolution problem that would reproduce `#177`'s symptom for a new reason.
_WEAK_RATING_ANCHOR = 600.0
_STRONG_RATING_ANCHOR = 2800.0


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
        """Return one embedding per timestep, shaped batch by sequence by width."""

        span = _STRONG_RATING_ANCHOR - _WEAK_RATING_ANCHOR
        weakness = (
            ((_STRONG_RATING_ANCHOR - rating.values.float()) / span)
            .clamp(0.0, 1.0)
            .unsqueeze(-1)
        )
        placed = weakness * self.weak + (1.0 - weakness) * self.strong
        return torch.where(rating.present.unsqueeze(-1), placed, self.unrated)


class SquareTokenEncoder(nn.Module):
    """Build the 64 square tokens each position is read as.

    Every token carries its own square's piece, the rule state that applies to
    the whole position, and the rating, so the spatial layers above this see a
    board whose geometry survived and a decision-maker whose strength is already
    part of the representation.
    """

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        embedding_dim = config.piece_embedding_dim
        self.piece_embedding = nn.Embedding(_PIECE_ID_COUNT, embedding_dim)
        self.side_embedding = nn.Embedding(_SIDE_TO_MOVE_COUNT, embedding_dim)
        self.castling_embedding = nn.Embedding(_CASTLING_RIGHTS_COUNT, embedding_dim)
        self.en_passant_embedding = nn.Embedding(EN_PASSANT_TOKEN_COUNT, embedding_dim)
        self.previous_action_embedding = nn.Embedding(
            PREVIOUS_ACTION_TOKEN_COUNT,
            config.action_embedding_dim,
        )
        self.square_identity = nn.Parameter(
            torch.empty(BOARD_SQUARE_COUNT, config.model_dim)
        )
        nn.init.normal_(self.square_identity, std=0.02)
        position_dim = 3 * embedding_dim + config.action_embedding_dim + 2
        # Two projections summed rather than one over a concatenation. A linear
        # map over concatenated inputs is exactly the sum of its parts, and the
        # position half is the same value on all 64 squares -- projecting it
        # once and broadcasting the result costs a fraction of replicating it
        # across the board first and is the same function.
        self.piece_projection = nn.Linear(embedding_dim, config.model_dim, bias=False)
        self.position_projection = nn.Linear(position_dim, config.model_dim)
        self.normalization = nn.LayerNorm(config.model_dim)
        self.rating_embedding = RatingEmbedding(config)

    def forward(self, batch: MoveModelBatch) -> Tensor:
        """Return tokens shaped batch by sequence by square by width."""

        inputs = batch.inputs
        pieces = self.piece_embedding(inputs.piece_ids)
        # A transform rather than a token vocabulary, so it stays here while the
        # two nullable inputs moved. Decision 0035 says why.
        rule_counts = torch.log1p(
            torch.stack(
                (
                    inputs.halfmove_clock.float(),
                    inputs.fullmove_number.float(),
                ),
                dim=-1,
            )
        )
        position = torch.cat(
            (
                self.side_embedding(inputs.side_to_move),
                self.castling_embedding(inputs.castling_rights),
                self.en_passant_embedding(inputs.en_passant_token),
                self.previous_action_embedding(inputs.previous_action_token),
                rule_counts,
            ),
            dim=-1,
        )
        tokens = self.piece_projection(pieces) + self.position_projection(
            position
        ).unsqueeze(-2)
        tokens = tokens + self.square_identity
        tokens = tokens + self.rating_embedding(inputs.target_rating).unsqueeze(-2)
        return cast(Tensor, self.normalization(tokens))


class GeometricAttentionBias(nn.Module):
    """Generate a per-head square-by-square attention bias from the position.

    Dot-product attention between square tokens has no idea which squares are
    adjacent, on a diagonal, or a knight's move apart, and a fixed positional
    encoding cannot say which of those relations matters in *this* position — a
    pinned bishop's diagonal is load-bearing and an empty one is not. This reads
    the whole board down to a small vector and mixes a learned set of
    64-by-64 templates from it, so the geometry each head attends along is
    chosen per position.

    The output layer starts at zero, so a fresh model is ordinary dot-product
    attention and the bias is something training adds rather than something it
    has to first undo.
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
        self.templates = nn.Linear(
            bias_dim,
            BOARD_SQUARE_COUNT * BOARD_SQUARE_COUNT,
        )
        nn.init.zeros_(self.templates.weight)
        nn.init.zeros_(self.templates.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return biases shaped position by head by square by square."""

        compressed = self.compression(hidden).flatten(-2)
        mixture = self.mixture_norm(
            self.board(compressed).unflatten(-1, (self.heads, -1))
        )
        return cast(
            Tensor,
            self.templates(mixture).unflatten(
                -1,
                (BOARD_SQUARE_COUNT, BOARD_SQUARE_COUNT),
            ),
        )


class MultiHeadAttention(nn.Module):
    """Multi-head attention over one axis, masked by whatever a subclass names.

    The two axes differ only in the mask they attend under, so that is the only
    thing a subclass supplies.
    """

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.model_dim // config.attention_heads
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.model_dim)
        # ``nn.Linear``'s own default is a generic fan-based one. A transformer
        # wants the projection drawn across all three of query, key, and value
        # at once, which is what this restates.
        nn.init.xavier_uniform_(self.qkv_projection.weight)
        nn.init.zeros_(self.qkv_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def masking(self, hidden: Tensor, dtype: torch.dtype) -> dict[str, Any]:
        """Return the mask keywords this axis attends under."""

        raise NotImplementedError

    def forward(self, hidden: Tensor) -> Tensor:
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
            **self.masking(hidden, query.dtype),
        )
        return cast(
            Tensor,
            self.output_projection(attended.transpose(1, 2).flatten(-2)),
        )


class SpatialAttention(MultiHeadAttention):
    """Attend between the squares of one position, along learned geometry."""

    def __init__(self, config: MoveModelConfig) -> None:
        super().__init__(config)
        self.geometric_bias = GeometricAttentionBias(config)

    def masking(self, hidden: Tensor, dtype: torch.dtype) -> dict[str, Any]:
        """Attend everywhere, tilted by the geometry read off this position."""

        return {"attn_mask": self.geometric_bias(hidden).to(dtype)}


class CausalSelfAttention(MultiHeadAttention):
    """Attend over earlier timesteps, stating causality as the flag it is.

    ``scaled_dot_product_attention`` reads ``is_causal`` as the mask rather than
    as a hint accompanying one. Flash and cuDNN attention refuse a non-null
    ``attn_mask``, so handing one over is what puts them out of reach — which is
    why this axis, the one whose length grows with the game, is the one kept
    free of a bias.
    """

    def masking(self, hidden: Tensor, dtype: torch.dtype) -> dict[str, Any]:
        """Attend backwards only, by the flag rather than a built mask."""

        return {"is_causal": True}


class ResidualBlock(nn.Module):
    """One pre-norm residual pair: the given attention, then a feed-forward.

    The spatial and causal stages differ only in which axis their attention
    reads, so the block around it is written once and handed the attention it
    should wrap.
    """

    def __init__(self, config: MoveModelConfig, attention: nn.Module) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = attention
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim),
        )
        # Dropout carries no state, so both residuals read the same module.
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        """Return the block's output for a whole batch of tokens."""

        hidden = hidden + self.residual_dropout(
            self.attention(self.attention_norm(hidden))
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
    """

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
        # Derived constants, not state: a pure function of the action
        # vocabulary, so they do not belong in a checkpoint.
        self.register_buffer(
            "move_square_slots",
            torch.tensor(square_slots, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "move_promotion_slots",
            torch.tensor(promotion_slots, dtype=torch.long),
            persistent=False,
        )

    def forward(self, squares: Tensor, hidden: Tensor) -> Tensor:
        """Return logits over the whole action vocabulary, moves then terminals."""

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
        return torch.cat((moves, self.terminal_projection(hidden)), dim=-1)


class CausalMoveModel(nn.Module):
    """Predict human action logits from exact state and causal trajectory."""

    position_table: Tensor

    def __init__(self, config: MoveModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoveModelConfig()
        self.square_encoder = SquareTokenEncoder(self.config)
        self.spatial_blocks = nn.ModuleList(
            ResidualBlock(self.config, SpatialAttention(self.config))
            for _ in range(self.config.spatial_layers)
        )
        self.spatial_norm = nn.LayerNorm(self.config.model_dim)
        # Pooling 64 normalized tokens leaves a vector an order of magnitude
        # shorter than the position features added to it next, so it is brought
        # back to their scale before the trunk rather than after.
        self.trajectory_norm = nn.LayerNorm(self.config.model_dim)
        self.transformer_blocks = nn.ModuleList(
            ResidualBlock(self.config, CausalSelfAttention(self.config))
            for _ in range(self.config.transformer_layers)
        )
        # A pre-norm block leaves its output unnormalized for whatever follows,
        # so the stack ends with one. Named rather than the last element of a
        # sequence, because a checkpoint key that moves with the layer count
        # cannot be read without also reading the configuration.
        self.transformer_norm = nn.LayerNorm(self.config.model_dim)
        self.decision_projection = nn.Linear(
            self.config.model_dim,
            self.config.model_dim,
        )
        self.decision_blocks = nn.ModuleList(
            ResidualBlock(self.config, SpatialAttention(self.config))
            for _ in range(self.config.decision_layers)
        )
        self.decision_norm = nn.LayerNorm(self.config.model_dim)
        self.action_head = SourceDestinationHead(self.config)
        # A derived constant, not state: a pure function of the configuration,
        # so it does not belong in a checkpoint.
        self.register_buffer(
            "position_table",
            _sinusoidal_table(
                self.config.maximum_context_plies,
                self.config.model_dim,
            ),
            persistent=False,
        )

    def forward(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits shaped batch by sequence by vocabulary.

        Every batch is validated by whichever factory built it, so nothing is
        rechecked here. Padded timesteps produce ordinary finite logits that no
        caller reads: the loss ignores them by target, and every scoring path
        gathers by the loss mask before looking.
        """

        squares, hidden = self.encode(batch)
        return cast(Tensor, self.action_head(self.decide(squares, hidden), hidden))

    def decide_at(self, batch: MoveModelBatch, decisions: Tensor) -> Tensor:
        """Return logits for one named ply per row, shaped batch by vocabulary.

        Serving reads a single decision out of each history, and the two stages
        after the trunk are the most expensive in the model — both run 64 tokens
        per ply, and the head builds a square-by-square board for each. Both are
        per-position, so taking the decision's row before them rather than after
        is the same arithmetic over one ply instead of the whole game.
        """

        squares, hidden = self.encode(batch)
        rows = torch.arange(squares.shape[0], device=squares.device)
        squares = squares[rows, decisions].unsqueeze(1)
        hidden = hidden[rows, decisions].unsqueeze(1)
        decided = self.action_head(self.decide(squares, hidden), hidden)
        return cast(Tensor, decided[:, 0])

    def encode(self, batch: MoveModelBatch) -> tuple[Tensor, Tensor]:
        """Return each position's encoded squares and the history over them."""

        declared = self.config.maximum_context_plies
        if batch.position_bound > declared:
            raise ValueError(
                f"batch reaches ply index {batch.position_bound - 1}, past the "
                f"{declared} plies this model declares as its context"
            )
        squares = self.encode_squares(batch)
        return squares, self.encode_trajectory(batch, squares)

    def encode_squares(self, batch: MoveModelBatch) -> Tensor:
        """Encode each position's squares, reading no position but its own."""

        return _spatially(
            self.square_encoder(batch), self.spatial_blocks, self.spatial_norm
        )

    def encode_trajectory(self, batch: MoveModelBatch, squares: Tensor) -> Tensor:
        """Encode rating-aware exact state and causal move history."""

        hidden = self.trajectory_norm(squares.mean(dim=-2))
        # A chunked selection carries ply indices past the row's own width.
        hidden = hidden + self.position_table[batch.ply_indices].to(hidden.dtype)
        # No key padding mask. Padding is right-aligned, so a real query attends
        # only to keys that are themselves real, and a padded query's output is
        # discarded by target and by loss mask downstream. Leaving it out also
        # removes the all-padding-row hazard a key padding mask carries.
        for block in self.transformer_blocks:
            hidden = block(hidden)
        return cast(Tensor, self.transformer_norm(hidden))

    def decide(self, squares: Tensor, hidden: Tensor) -> Tensor:
        """Read the history feature back onto the squares the head scores."""

        decision = squares + self.decision_projection(hidden).unsqueeze(-2)
        return _spatially(decision, self.decision_blocks, self.decision_norm)

    def identity(self) -> dict[str, object]:
        """Return compatibility metadata for future runs and checkpoints."""

        return model_identity(self.config)


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
        "name": "anthro-causal-move-model",
        # Version 6 is the deliberate architecture 0066 records: square tokens,
        # rating in the input representation, and a source-destination move
        # head. None of the three can read a version 5 checkpoint.
        "version": 6,
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


def _spatially(
    tokens: Tensor,
    blocks: nn.ModuleList,
    normalization: nn.Module,
) -> Tensor:
    """Run a stack over each position's squares, batch and ply folded together.

    A spatial stage reads one position at a time, so the two leading dimensions
    are collapsed into the batch for the whole stack and restored after it.
    """

    positions = tokens.flatten(0, 1)
    for block in blocks:
        positions = block(positions)
    return cast(Tensor, normalization(positions).unflatten(0, tokens.shape[:2]))


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
        choice = (
            _PROMOTION_CHOICE_COUNT
            if move.promotion is None
            else move.promotion - chess.KNIGHT
        )
        promotion_slots.append(move.to_square * (_PROMOTION_CHOICE_COUNT + 1) + choice)
    return tuple(square_slots), tuple(promotion_slots)


def _sinusoidal_table(length: int, dimension: int) -> Tensor:
    dtype = torch.get_default_dtype()
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=dtype) * (-math.log(10_000.0) / dimension)
    )
    angles = torch.arange(length, dtype=dtype).unsqueeze(-1) * frequencies
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
