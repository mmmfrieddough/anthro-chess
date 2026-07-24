"""Protocol-independent chess game sessions and legal action selection."""

from __future__ import annotations

import logging
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


class GameSession:
    """Own exact game state and choose actions for one controlled color."""

    def __init__(
        self,
        runner: ActionModelRunner,
        *,
        controlled_color: chess.Color,
        config: RuntimeConfig | None = None,
    ) -> None:
        if type(controlled_color) is not bool:
            raise TypeError("controlled_color must be chess.WHITE or chess.BLACK")
        self._runner = runner
        self.controlled_color = controlled_color
        self.config = config or RuntimeConfig()
        self._generator = torch.Generator(device="cpu")
        self._board = chess.Board()
        self._resigned_by: chess.Color | None = None
        self.reset()

    @property
    def board(self) -> chess.Board:
        """Return a defensive copy of the canonical board."""

        return self._board.copy(stack=True)

    @property
    def move_history(self) -> tuple[chess.Move, ...]:
        """Return every observed move from both players."""

        return tuple(self._board.move_stack)

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
        """Replace game-local state after validating the complete move history."""

        try:
            board = chess.Board(initial_fen)
        except ValueError as error:
            raise SessionStateError(f"invalid initial position: {error}") from error
        for ply_index, move in enumerate(tuple(moves)):
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
        self._board = board
        self._resigned_by = None
        self._generator.manual_seed(self.config.seed)
        logger.debug(
            "Reset game session for controlled color %s with %s observed plies",
            "white" if self.controlled_color == chess.WHITE else "black",
            len(board.move_stack),
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
        """Select, apply, and return one valid action for the controlled color."""

        if self.is_terminal:
            raise SessionStateError("cannot choose an action in a terminal game")
        if self._board.turn != self.controlled_color:
            color = "white" if self.controlled_color == chess.WHITE else "black"
            raise SessionStateError(
                f"cannot choose an action when it is not {color}'s turn"
            )

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
            self._resigned_by = self.controlled_color
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
