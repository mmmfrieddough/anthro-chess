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
from anthro_chess.data import DecisionContext, DecisionHistory, EncodingError
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


@dataclass(frozen=True)
class SelectionPolicy:
    """What the policy said about one selected action.

    The distribution is over the actions the position enabled, not over the
    raw vocabulary, so these describe the decision that was actually available
    rather than the model's unmasked preference. It is untempered; see
    :meth:`describe` for why.

    Reporting them here is what lets a benchmark rollout and a game
    reconstructed from a live session carry the same per-decision quantities:
    both go through this one selection path.
    """

    enabled_action_count: int
    selected_probability: float
    selected_rank: int
    preferred_action_id: int
    preferred_probability: float

    @classmethod
    def describe(
        cls,
        candidate_logits: Tensor,
        enabled_ids: tuple[int, ...],
        candidate_index: int,
    ) -> SelectionPolicy:
        """Describe one enabled action under the model's own policy.

        The reported probabilities come from the untempered distribution over
        enabled actions rather than from the tempered one a draw would use. The
        temperature is a dial recorded beside the decision, so reading these
        through it would make the model's confidence move whenever the dial
        did, and would collapse to a point mass at temperature zero exactly
        where the model's confidence in its greedy choice is the interesting
        quantity.

        Sampling and after-the-fact scoring both come here, so a decomposition
        of a game the runtime did not originate describes it in the same terms
        the runtime reported at the time.
        """

        probabilities = torch.softmax(candidate_logits, dim=0)
        preferred_index = int(torch.argmax(candidate_logits).item())
        selected_logit = candidate_logits[candidate_index]
        rank = int((candidate_logits > selected_logit).sum().item()) + 1
        return cls(
            enabled_action_count=len(enabled_ids),
            selected_probability=float(probabilities[candidate_index].item()),
            selected_rank=rank,
            preferred_action_id=enabled_ids[preferred_index],
            preferred_probability=float(probabilities[preferred_index].item()),
        )


@dataclass(frozen=True)
class ActionDecision:
    """One selected action together with the policy that produced it."""

    action: GameAction
    policy: SelectionPolicy


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
        self._history = DecisionHistory()
        self._resigned_by: chess.Color | None = None
        self._resolved_seed = 0
        self.reset(initial_fen=initial_fen, moves=moves)

    @property
    def _board(self) -> chess.Board:
        """Return the live canonical board the encoded history owns."""

        return self._history.board

    @property
    def board(self) -> chess.Board:
        """Return a defensive copy of the canonical board."""

        return self._history.board.copy(stack=True)

    @property
    def initial_fen(self) -> str:
        """Return the FEN the current game history is rooted at."""

        return self._history.initial_fen

    @property
    def move_history(self) -> tuple[chess.Move, ...]:
        """Return every observed move from both players."""

        return self._history.moves

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
        """Start a new game and a new random stream after validating history.

        A new game shares nothing with the old one, so the encoded history is
        rebuilt rather than resynchronized. Encoding it is the validation, and
        the plies it produces are the ones the first decision reads, so nothing
        is built here only to be built again when the model is asked to move.
        """

        try:
            history = DecisionHistory(initial_fen=initial_fen, moves=moves)
        except EncodingError as error:
            raise SessionStateError(str(error)) from error
        self._history = history
        self._resigned_by = None
        self._begin_random_stream()
        logger.debug(
            "Reset game session with %s observed plies",
            len(history.moves),
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
        same_root = self._history.initial_fen == initial_fen
        current_plies = len(self._history.moves)
        try:
            reused = self._history.synchronize(initial_fen=initial_fen, moves=moves)
        except EncodingError as error:
            raise SessionStateError(str(error)) from error
        # Keeping every ply of the previous history is exactly what makes an
        # update append-only; anything less replaced part of the game.
        replaced = not (same_root and reused == current_plies)

        self._resigned_by = None
        logger.debug(
            "Synced position: %s, reused %s of %s plies",
            "replaced" if replaced else "append-only",
            reused,
            len(moves),
        )
        return PositionSync(
            total_plies=len(moves),
            reused_prefix_plies=reused,
            replaced=replaced,
        )

    def reseed(self) -> None:
        """Establish the next random stream from the current seed policy."""

        self._begin_random_stream()

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
        self._history.push(move)

    def choose_action(self) -> GameAction:
        """Select, apply, and return one valid action for the player to move."""

        return self.decide().action

    def decide(self) -> ActionDecision:
        """Select and apply one action, reporting the policy behind it."""

        logits, enabled_ids = self._policy_inputs()
        action_id, policy = self._sample_action(logits, enabled_ids)

        if action_id == RESIGNATION_ACTION_ID:
            self._resigned_by = self._board.turn
            logger.debug("Selected resignation action")
            return ActionDecision(action=ResignationAction(), policy=policy)

        move = decode_move(action_id)
        if move not in self._board.legal_moves:
            raise ActionSelectionError(
                "selected move is not legal in the current position"
            )
        self._history.push(move)
        logger.debug("Selected and applied move action %s", action_id)
        return ActionDecision(
            action=MoveAction(action_id=action_id, move=move),
            policy=policy,
        )

    def score_action(self, action_id: int) -> SelectionPolicy:
        """Describe what the policy says about one enabled action right now.

        Reads the model without deciding anything: the board is unchanged and
        the random stream is untouched, because no draw is made. This is how a
        game the runtime did not originate is decomposed — a manually played
        game reconstructed from a log, or any recorded game whose per-decision
        policy quantities were never stored — through the same selection path
        that would have produced the decision.
        """

        if type(action_id) is not int:
            raise TypeError("action id must be an int")
        logits, enabled_ids = self._policy_inputs()
        try:
            candidate_index = enabled_ids.index(action_id)
        except ValueError:
            raise ActionSelectionError(
                f"action {action_id} is not enabled in the current position"
            ) from None
        candidate_logits = logits[torch.tensor(enabled_ids, dtype=torch.long)]
        return SelectionPolicy.describe(candidate_logits, enabled_ids, candidate_index)

    def _policy_inputs(self) -> tuple[Tensor, tuple[int, ...]]:
        """Return validated action logits and the actions this position enables.

        Shared by deciding and by scoring, so re-scoring a recorded decision
        reads the same encoded history, the same legal mask, and the same logit
        validation the live path does.
        """

        if self.is_terminal:
            raise SessionStateError("cannot choose an action in a terminal game")

        try:
            context = self._history.context(target_rating=self.config.target_rating)
        except EncodingError as error:
            raise SessionStateError(
                f"cannot build decision context: {error}"
            ) from error
        logits = self._validate_logits(self._runner.predict(context))
        enabled_ids = legal_action_ids(
            self._board,
            include_resignation=self.config.resignation_enabled,
        )
        return logits, enabled_ids

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

    def _sample_action(
        self,
        logits: Tensor,
        enabled_ids: tuple[int, ...],
    ) -> tuple[int, SelectionPolicy]:
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
        return (
            enabled_ids[candidate_index],
            SelectionPolicy.describe(candidate_logits, enabled_ids, candidate_index),
        )
