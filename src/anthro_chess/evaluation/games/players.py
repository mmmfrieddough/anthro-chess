"""The player configurations the generation harness pairs.

Every consumer of generated games differs only in who sits in the two seats: a
self-play rollout puts the same configuration in both, a rating ladder puts two
ratings against each other, an engine anchor puts an external process opposite
the model, and a robustness rollout puts a uniform-random opponent there. So a
player is an interface rather than a mode flag, and the harness never learns
which kind it is driving.

A player is a configuration that outlives a suite; a seat is its per-game
state. Splitting them is what lets one loaded checkpoint or one engine process
serve every game while each game still gets its own random stream.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import chess
from chess import engine as chess_engine
from torch import Tensor

from anthro_chess.chess import RESIGNATION_ACTION_ID, encode_move, legal_action_ids
from anthro_chess.data import DecisionContext
from anthro_chess.evaluation.games.records import DecisionPolicy, SeatRecord
from anthro_chess.evaluation.results.records import CheckpointReference
from anthro_chess.runtime import (
    ActionModelRunner,
    BatchedActionModelRunner,
    DecisionRuntimeError,
    GameSession,
    MoveAction,
    RuntimeConfig,
)

logger = logging.getLogger(__name__)


class PlayerError(ValueError):
    """Raised when a seat cannot produce a valid action."""


@dataclass(frozen=True)
class DecisionRequest:
    """The position one seat is being asked to move in.

    The board carries its own move stack, so a seat that needs the trajectory
    reads it from there rather than from a parallel history that could drift
    out of step with the position.
    """

    board: chess.Board
    initial_position: str
    ply_index: int

    @property
    def move_history(self) -> tuple[chess.Move, ...]:
        """Return every move played since the game's initial position."""

        return tuple(self.board.move_stack)


@dataclass(frozen=True)
class SeatDecision:
    """One action a seat chose, with the policy behind it when it has one."""

    action_id: int
    policy: DecisionPolicy | None = None


class PlayerSeat(Protocol):
    """One player's state for one game."""

    def decide(self, request: DecisionRequest) -> SeatDecision:
        """Choose one enabled action in the requested position."""

    def close(self) -> None:
        """Release any per-game state."""


class GamePlayer(Protocol):
    """One player configuration, reused across the games of a suite."""

    @property
    def supports_concurrent_games(self) -> bool:
        """Return whether several of this player's seats may be open at once.

        A player backed by shared state that only tracks one game at a time,
        such as a single engine process, says no and the harness plays its
        suites one game at a time.
        """

    def seat_record(self, *, seed: int) -> SeatRecord:
        """Return the resolved configuration this seat plays a game under."""

    def seat(self, *, seed: int) -> PlayerSeat:
        """Open per-game state for one seat with its own random stream."""

    def close(self) -> None:
        """Release any state that outlives a single game."""


@runtime_checkable
class BatchingGamePlayer(GamePlayer, Protocol):
    """A player that can resolve several of its seats' decisions together.

    Separate from :class:`GamePlayer` because most players have nothing to
    gain from it: a uniform-random seat is already free, and an external
    engine decides in its own process. A player that does not implement this
    is asked one seat at a time, so batching is an optimization the harness
    offers rather than a requirement it imposes.
    """

    def decide_batch(
        self,
        pending: Sequence[tuple[PlayerSeat, DecisionRequest]],
    ) -> tuple[SeatDecision, ...]:
        """Choose one action for each pending seat, in the order given."""


class ModelPlayer:
    """A checkpoint playing through the ordinary decision runtime.

    The seat drives a real :class:`GameSession` rather than reimplementing
    selection, so a benchmark measures the policy the engine actually plays
    and inherits the session's position synchronization, legal masking, and
    seeding rules unchanged.
    """

    def __init__(
        self,
        runner: ActionModelRunner,
        *,
        label: str,
        config: RuntimeConfig | None = None,
        checkpoint: CheckpointReference | None = None,
    ) -> None:
        self._runner = runner
        self._label = label
        self._config = config or RuntimeConfig()
        self._checkpoint = checkpoint

    @property
    def config(self) -> RuntimeConfig:
        """Return the runtime settings every seat of this player uses."""

        return self._config

    @property
    def supports_concurrent_games(self) -> bool:
        """Report that seats are independent sessions over a shared runner."""

        return True

    def seat_record(self, *, seed: int) -> SeatRecord:
        """Return this player's resolved configuration for one game."""

        return SeatRecord(
            kind="model",
            label=self._label,
            seed=seed,
            configuration=self._config.model_dump(mode="json", exclude={"seed"}),
            checkpoint=self._checkpoint,
        )

    def seat(self, *, seed: int) -> ModelSeat:
        """Open one seeded game session for this configuration."""

        seeded = self._config.model_copy(update={"seed": seed})
        return ModelSeat(self._runner, config=seeded)

    def decide_batch(
        self,
        pending: Sequence[tuple[PlayerSeat, DecisionRequest]],
    ) -> tuple[SeatDecision, ...]:
        """Resolve several of this player's pending decisions in one pass.

        Every seat is synchronized and encoded first, the runner is asked once
        for the whole wave, and each seat then samples from its own logits
        through its own session. Sampling stays per seat, so which games
        happened to share a pass does not enter any game's random stream.
        """

        if not pending:
            return ()
        seats = tuple(_model_seat(seat) for seat, _ in pending)
        contexts = tuple(
            seat.prepare(request)
            for seat, (_, request) in zip(seats, pending, strict=True)
        )
        if isinstance(self._runner, BatchedActionModelRunner) and len(contexts) > 1:
            try:
                logits = self._runner.predict_batch(contexts)
            except ValueError as error:
                raise PlayerError(f"model seats cannot decide: {error}") from error
            if len(logits) != len(contexts):
                raise PlayerError("model runner returned the wrong number of decisions")
        else:
            logits = tuple(self._runner.predict(context) for context in contexts)
        return tuple(seat.resolve(row) for seat, row in zip(seats, logits, strict=True))

    def close(self) -> None:
        """Leave the shared runner alone; its lifetime is the caller's."""


def _model_seat(seat: PlayerSeat) -> ModelSeat:
    if not isinstance(seat, ModelSeat):
        raise PlayerError("a model player can only resolve its own seats")
    return seat


class ModelSeat:
    """One game's session for a model player."""

    def __init__(self, runner: ActionModelRunner, *, config: RuntimeConfig) -> None:
        self._runner = runner
        self._session = GameSession(runner, config=config)

    def decide(self, request: DecisionRequest) -> SeatDecision:
        """Synchronize the session to the position and select one action."""

        return self.resolve(self._runner.predict(self.prepare(request)))

    def prepare(self, request: DecisionRequest) -> DecisionContext:
        """Synchronize the session and return the context to predict from."""

        try:
            self._session.sync_position(
                initial_fen=request.initial_position,
                moves=request.move_history,
            )
            return self._session.decision_context()
        except DecisionRuntimeError as error:
            raise PlayerError(f"model seat cannot decide: {error}") from error

    def resolve(self, logits: Tensor) -> SeatDecision:
        """Select and apply one action from this seat's own action logits."""

        try:
            decision = self._session.decide_from_logits(logits)
        except DecisionRuntimeError as error:
            raise PlayerError(f"model seat cannot decide: {error}") from error
        action_id = (
            decision.action.action_id
            if isinstance(decision.action, MoveAction)
            else RESIGNATION_ACTION_ID
        )
        return SeatDecision(
            action_id=action_id,
            policy=DecisionPolicy.from_selection(decision.policy),
        )

    def close(self) -> None:
        """Drop the session; nothing outside this game holds it."""


class RandomPlayer:
    """A player choosing uniformly among the position's legal moves.

    This is the control arm for robustness work rather than a strength
    baseline. It never resigns, because a random resignation would end games
    at a rate that has nothing to do with the position.
    """

    def __init__(self, *, label: str = "uniform-random") -> None:
        self._label = label

    @property
    def supports_concurrent_games(self) -> bool:
        """Report that each seat owns its own random stream and nothing else."""

        return True

    def seat_record(self, *, seed: int) -> SeatRecord:
        """Return this player's resolved configuration for one game."""

        return SeatRecord(kind="random", label=self._label, seed=seed)

    def seat(self, *, seed: int) -> RandomSeat:
        """Open one seeded uniform-random seat."""

        return RandomSeat(seed=seed)

    def close(self) -> None:
        """No state outlives a game."""


class RandomSeat:
    """One game's random stream for a uniform-random player."""

    def __init__(self, *, seed: int) -> None:
        self._random = random.Random(seed)

    def decide(self, request: DecisionRequest) -> SeatDecision:
        """Draw one legal move uniformly from a reproducible stream.

        Candidates are ordered by action id rather than by however the board
        happens to generate them, so a recorded seed reproduces the same game
        regardless of move-generation order.

        No policy is reported. Uniform selection has no preferred action, and
        recording a rank of one for every legal move would read like a
        preference the player does not have.
        """

        candidates = legal_action_ids(request.board, include_resignation=False)
        if not candidates:
            raise PlayerError("the requested position has no legal moves")
        return SeatDecision(action_id=self._random.choice(candidates))

    def close(self) -> None:
        """No state outlives a game."""


class ExternalEngine(Protocol):
    """The external process surface an engine seat drives.

    Deliberately narrower than a UCI client library: a move for a position, a
    new-game boundary, and a shutdown. Anything a test needs to stand in for is
    then three methods rather than a protocol implementation.
    """

    def new_game(self) -> None:
        """Tell the engine the next position belongs to a new game."""

    def play(self, board: chess.Board) -> chess.Move:
        """Return the engine's move for one position."""

    def close(self) -> None:
        """Shut the engine down."""


class ExternalEnginePlayer:
    """An external UCI engine occupying one seat.

    One process serves every game of a suite; seats mark the game boundaries.
    The engine owns its own determinism, so the harness records the seat's seed
    as absent rather than implying a reproducibility it cannot provide.
    """

    def __init__(
        self,
        engine: ExternalEngine,
        *,
        label: str,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self._engine = engine
        self._label = label
        self._configuration = dict(configuration or {})

    @property
    def supports_concurrent_games(self) -> bool:
        """Report that one process cannot hold several games at once.

        Seats mark game boundaries on a shared process, so overlapping games
        would leave the engine believing it is still in the previous one. A
        suite with an engine seat therefore plays one game at a time.
        """

        return False

    def seat_record(self, *, seed: int) -> SeatRecord:
        """Return this player's resolved configuration for one game."""

        return SeatRecord(
            kind="external-engine",
            label=self._label,
            seed=None,
            configuration=dict(self._configuration),
        )

    def seat(self, *, seed: int) -> ExternalEngineSeat:
        """Mark a new game on the shared process and return its seat."""

        self._engine.new_game()
        return ExternalEngineSeat(self._engine)

    def close(self) -> None:
        """Shut the shared engine process down."""

        self._engine.close()


class ExternalEngineSeat:
    """One game's boundary on a shared external engine process."""

    def __init__(self, engine: ExternalEngine) -> None:
        self._engine = engine

    def decide(self, request: DecisionRequest) -> SeatDecision:
        """Ask the engine for a move and encode it as an action.

        No policy is reported: an external engine exposes a move, not a
        distribution over this project's action vocabulary.
        """

        try:
            move = self._engine.play(request.board.copy(stack=True))
        except (OSError, RuntimeError, ValueError) as error:
            if isinstance(error, PlayerError):
                raise
            raise PlayerError(f"external engine failed to move: {error}") from error
        if not isinstance(move, chess.Move):
            raise PlayerError("external engine returned something other than a move")
        if move not in request.board.legal_moves:
            raise PlayerError(
                f"external engine returned illegal move {move.uci()} in the "
                "requested position"
            )
        return SeatDecision(action_id=encode_move(move))

    def close(self) -> None:
        """Leave the shared process running for the next game."""


class UciEngineProcess:
    """A ``python-chess`` engine driven under a wall-clock-free search limit.

    Search is bounded by depth or nodes rather than by time. A time limit would
    make results depend on how loaded the machine was, which is the one thing a
    benchmark comparing checkpoints across months cannot afford.
    """

    def __init__(
        self,
        engine: chess_engine.SimpleEngine,
        *,
        depth: int | None = None,
        nodes: int | None = None,
    ) -> None:
        if (depth is None) == (nodes is None):
            raise PlayerError("an engine seat needs exactly one of depth or nodes")
        self._engine = engine
        self._depth = depth
        self._nodes = nodes
        self._game = object()

    @classmethod
    def launch(
        cls,
        command: Sequence[str],
        *,
        depth: int | None = None,
        nodes: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> UciEngineProcess:
        """Start an external UCI engine and configure it once."""

        try:
            engine = chess_engine.SimpleEngine.popen_uci(list(command))
            if options:
                engine.configure(dict(options))
        except (OSError, chess_engine.EngineError) as error:
            raise PlayerError(f"cannot start engine {command!r}: {error}") from error
        logger.info("Started external engine %s", command[0])
        return cls(engine, depth=depth, nodes=nodes)

    @property
    def limit_record(self) -> dict[str, Any]:
        """Return the search limit for the seat's resolved configuration."""

        return {"depth": self._depth, "nodes": self._nodes}

    def new_game(self) -> None:
        """Mark a game boundary so the engine clears its per-game state.

        ``python-chess`` sends ``ucinewgame`` when the game token it is given
        changes, so replacing the token is how a new game is announced.
        """

        self._game = object()

    def play(self, board: chess.Board) -> chess.Move:
        """Return the engine's move under the configured search limit."""

        result = self._engine.play(
            board,
            chess_engine.Limit(depth=self._depth, nodes=self._nodes),
            game=self._game,
        )
        if result.move is None:
            raise PlayerError("external engine declined to move")
        return result.move

    def close(self) -> None:
        """Quit the engine process."""

        self._engine.quit()
