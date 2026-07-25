"""Protocol-independent chess game sessions and legal action selection."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import chess
import torch
from torch import Tensor

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    RESIGNATION_ACTION_ID,
    decode_move,
    legal_action_ids,
)
from anthro_chess.data import DecisionContext, EncodingError, build_decision_context
from anthro_chess.runtime.config import RuntimeConfig

logger = logging.getLogger(__name__)


class DecisionRuntimeError(ValueError):
    """Base error for invalid game state or model decisions."""


class SessionStateError(DecisionRuntimeError):
    """Raised when a requested session transition is invalid."""


class ActionSelectionError(DecisionRuntimeError):
    """Raised when model output cannot produce a valid enabled action."""


class ActionModelRunner(Protocol):
    """The model-runner surface used by the decision runtime."""

    def predict(self, context: DecisionContext) -> Tensor:
        """Return raw logits over the shared action vocabulary."""


@dataclass(frozen=True)
class MoveAction:
    """A legal move selected and applied by the session."""

    action_id: int
    move: chess.Move


@dataclass(frozen=True)
class ResignationAction:
    """A learned resignation selected and applied by the session."""

    action_id: int = RESIGNATION_ACTION_ID


GameAction: TypeAlias = MoveAction | ResignationAction

# Fresh streams draw a positive seed that fits a signed 64-bit integer so it is
# safe to log, reproduce, and pass to ``torch.Generator.manual_seed``.
_FRESH_SEED_BITS = 63


def _draw_fresh_seed() -> int:
    """Draw one fresh per-game seed from operating-system entropy."""

    return secrets.randbits(_FRESH_SEED_BITS)


@dataclass(frozen=True)
class PositionSync:
    """How one position synchronization resolved against prior game state."""

    total_plies: int
    reused_prefix_plies: int
    replaced: bool

    @property
    def appended_plies(self) -> int:
        """Return plies past the reusable prefix that need fresh encoding."""

        return self.total_plies - self.reused_prefix_plies


class GameSession:
    """Own exact game state and choose an action for the player to move.

    A session has no controlled color. Exact board state already identifies the
    side to move, and the target rating conditions that decision rather than a
    player, so one session serves either side of a game. Callers that assign
    colors, such as a game loop that alternates with a human, own that mapping
    and decide when to ask for a move.
    """

    def __init__(
        self,
        runner: ActionModelRunner,
        *,
        config: RuntimeConfig | None = None,
        initial_fen: str = chess.STARTING_FEN,
        moves: Sequence[chess.Move] = (),
    ) -> None:
        self._runner = runner
        self.config = config or RuntimeConfig()
        self._generator = torch.Generator(device="cpu")
        self._board = chess.Board()
        self._initial_fen = chess.STARTING_FEN
        self._resigned_by: chess.Color | None = None
        self._reusable_prefix_plies = 0
        self._resolved_seed = 0
        self.reset(initial_fen=initial_fen, moves=moves)

    @property
    def board(self) -> chess.Board:
        """Return a defensive copy of the canonical board."""

        return self._board.copy(stack=True)

    @property
    def initial_fen(self) -> str:
        """Return the FEN the current game history is rooted at."""

        return self._initial_fen

    @property
    def move_history(self) -> tuple[chess.Move, ...]:
        """Return every observed move from both players."""

        return tuple(self._board.move_stack)

    @property
    def reusable_prefix_plies(self) -> int:
        """Return plies whose encoded history survived the last sync."""

        return self._reusable_prefix_plies

    @property
    def resolved_seed(self) -> int:
        """Return the seed that established the active random stream."""

        return self._resolved_seed

    @property
    def resigned_by(self) -> chess.Color | None:
        """Return the color that resigned through this session, if any."""

        return self._resigned_by

    @property
    def is_terminal(self) -> bool:
        """Return whether no further session actions are valid."""

        return self._resigned_by is not None or self._board.is_game_over()

    def reset(
        self,
        *,
        initial_fen: str = chess.STARTING_FEN,
        moves: Sequence[chess.Move] = (),
    ) -> None:
        """Start a new game and a new random stream after validating history."""

        board = self._reconstruct_and_validate(initial_fen, tuple(moves))
        self._board = board
        self._initial_fen = initial_fen
        self._resigned_by = None
        self._reusable_prefix_plies = 0
        self._begin_random_stream()
        logger.debug(
            "Reset game session with %s observed plies",
            len(board.move_stack),
        )

    def sync_position(
        self,
        *,
        initial_fen: str = chess.STARTING_FEN,
        moves: Sequence[chess.Move] = (),
    ) -> PositionSync:
        """Advance to a target position without disturbing the random stream.

        An append-only history extends the live board and preserves the encoded
        prefix. A new root, takeback, or divergent history atomically replaces
        the board and invalidates the cached prefix past the divergence point.
        The active random stream is never reseeded or rewound here; only a new
        game through :meth:`reset` or :meth:`reseed` establishes a new stream.
        """

        moves = tuple(moves)
        target = self._reconstruct_and_validate(initial_fen, moves)
        current = tuple(self._board.move_stack)
        same_root = initial_fen == self._initial_fen
        prefix = _common_prefix_length(current, moves) if same_root else 0

        self._resigned_by = None
        if same_root and prefix == len(current) and len(moves) >= len(current):
            for move in moves[len(current) :]:
                self._board.push(move)
            self._reusable_prefix_plies = len(current)
            replaced = False
        else:
            self._board = target
            self._initial_fen = initial_fen
            self._reusable_prefix_plies = prefix
            replaced = True
        logger.debug(
            "Synced position: %s, reused %s of %s plies",
            "replaced" if replaced else "append-only",
            self._reusable_prefix_plies,
            len(moves),
        )
        return PositionSync(
            total_plies=len(moves),
            reused_prefix_plies=self._reusable_prefix_plies,
            replaced=replaced,
        )

    def reseed(self) -> None:
        """Establish the next random stream from the current seed policy."""

        self._begin_random_stream()

    def _reconstruct_and_validate(
        self,
        initial_fen: str,
        moves: tuple[chess.Move, ...],
    ) -> chess.Board:
        try:
            board = chess.Board(initial_fen)
        except ValueError as error:
            raise SessionStateError(f"invalid initial position: {error}") from error
        for ply_index, move in enumerate(moves):
            if not isinstance(move, chess.Move):
                raise TypeError(f"move at ply {ply_index} must be a chess.Move")
            if move not in board.legal_moves:
                raise SessionStateError(
                    f"move history is illegal at ply {ply_index}: {move.uci()}"
                )
            board.push(move)
        try:
            build_decision_context(
                board,
                tuple(board.move_stack),
                target_rating=self.config.target_rating,
            )
        except EncodingError as error:
            raise SessionStateError(f"invalid move history: {error}") from error
        return board

    def _begin_random_stream(self) -> None:
        seed = self.config.seed
        explicit = seed is not None
        if seed is None:
            seed = _draw_fresh_seed()
        self._resolved_seed = seed
        self._generator.manual_seed(seed)
        logger.debug(
            "Began %s random stream with resolved seed %s",
            "explicit" if explicit else "fresh",
            seed,
        )

    def apply_move(self, move: chess.Move) -> None:
        """Apply one observed legal move from either player."""

        if not isinstance(move, chess.Move):
            raise TypeError("move must be a chess.Move")
        if self.is_terminal:
            raise SessionStateError("cannot apply a move to a terminal game")
        if move not in self._board.legal_moves:
            raise SessionStateError(
                f"cannot apply illegal move {move.uci()} in the current position"
            )
        self._board.push(move)

    def choose_action(self) -> GameAction:
        """Select, apply, and return one valid action for the player to move."""

        if self.is_terminal:
            raise SessionStateError("cannot choose an action in a terminal game")

        try:
            context = build_decision_context(
                self._board,
                self.move_history,
                target_rating=self.config.target_rating,
            )
        except EncodingError as error:
            raise SessionStateError(
                f"cannot build decision context: {error}"
            ) from error
        logits = self._validate_logits(self._runner.predict(context))
        enabled_ids = legal_action_ids(
            self._board,
            include_resignation=self.config.resignation_enabled,
        )
        action_id = self._sample_action(logits, enabled_ids)

        if action_id == RESIGNATION_ACTION_ID:
            self._resigned_by = self._board.turn
            logger.debug("Selected resignation action")
            return ResignationAction()

        move = decode_move(action_id)
        if move not in self._board.legal_moves:
            raise ActionSelectionError(
                "selected move is not legal in the current position"
            )
        self._board.push(move)
        logger.debug("Selected and applied move action %s", action_id)
        return MoveAction(action_id=action_id, move=move)

    @staticmethod
    def _validate_logits(logits: object) -> Tensor:
        if not isinstance(logits, Tensor):
            raise ActionSelectionError("model runner must return a torch.Tensor")
        if logits.shape != (ACTION_VOCABULARY_SIZE,):
            raise ActionSelectionError(
                "model runner returned an invalid action-logit shape"
            )
        observed = logits.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(observed).all():
            raise ActionSelectionError("model runner returned non-finite action logits")
        return observed

    def _sample_action(self, logits: Tensor, enabled_ids: tuple[int, ...]) -> int:
        if not enabled_ids:
            raise ActionSelectionError("the current position has no enabled actions")
        candidate_ids = torch.tensor(enabled_ids, dtype=torch.long)
        candidate_logits = logits[candidate_ids]
        if self.config.temperature == 0.0:
            candidate_index = int(torch.argmax(candidate_logits).item())
        else:
            probabilities = torch.softmax(
                candidate_logits / self.config.temperature,
                dim=0,
            )
            if (
                not torch.isfinite(probabilities).all()
                or probabilities.sum().item() <= 0.0
            ):
                raise ActionSelectionError(
                    "enabled action logits cannot form a sampling distribution"
                )
            candidate_index = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self._generator,
                ).item()
            )
        return enabled_ids[candidate_index]


def _common_prefix_length(
    current: tuple[chess.Move, ...],
    target: tuple[chess.Move, ...],
) -> int:
    length = 0
    for existing, incoming in zip(current, target, strict=False):
        if existing != incoming:
            break
        length += 1
    return length
