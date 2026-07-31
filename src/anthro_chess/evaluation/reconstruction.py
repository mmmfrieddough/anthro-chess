"""Recovering the decisions of a played game from a runtime debug log.

A game played through a chess GUI is the one source of decisions no benchmark
produces: real opponent moves, real settings, and the positions a person
actually steered the engine into. The UCI adapter records accepted position
snapshots and its own decisions as versioned DEBUG events for exactly this
reason; this module reads them back.

What comes out is a list of decisions, each with the exact history it was made
from and the settings in force at the time, which is what
:func:`anthro_chess.evaluation.decisions.score_played_decisions` needs. It is
deliberately not a game record: a session log can end mid-game, and inventing
an outcome for it would put a fabricated termination into the one format the
termination benchmarks read.

Takebacks and replaced histories need no special handling. The events are
replayed in the order the engine accepted them, so whatever history was in force
when a decision was made is the history that decision carries.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from anthro_chess.evaluation.decisions import (
    DecisionDecomposition,
    DecisionDecompositionError,
    PlayedDecision,
    score_played_decisions,
    summarize_decisions,
)
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerConfig
from anthro_chess.inference.runner import ModelRunnerError
from anthro_chess.interfaces.uci import (
    UCI_GAME_EVENT_PREFIX,
    UCI_GAME_EVENT_SCHEMA,
)
from anthro_chess.runtime import RuntimeConfig

logger = logging.getLogger(__name__)


class ReconstructionError(ValueError):
    """Raised when a log does not describe a game that can be replayed."""


@dataclass(frozen=True)
class ReconstructedGame:
    """One game a logged process played, and the decisions it made in it.

    ``moves`` is the history as it stood when the log ended, which for an
    abandoned game is a partial game rather than a finished one. ``decisions``
    covers only the plies the engine itself chose; the opponent's moves arrive
    as accepted positions and are history rather than decisions.
    """

    session_id: str
    game_index: int
    initial_fen: str
    moves: tuple[chess.Move, ...]
    decisions: tuple[PlayedDecision, ...]

    @property
    def game_id(self) -> str:
        """Return the identity this game is reported under."""

        return _game_id(self.session_id, self.game_index)


def reconstruct_uci_games(lines: Iterable[str]) -> tuple[ReconstructedGame, ...]:
    """Return every game a UCI debug log describes, in the order they started.

    Lines that carry no game event are ignored, so an ordinary application log
    with interleaved diagnostics can be passed whole. A line that carries an
    event this build cannot read is an error rather than a skip: silently
    dropping events would produce a shorter game that still looks valid.
    """

    states: dict[tuple[str, int], _GameState] = {}
    order: list[tuple[str, int]] = []
    for event in _events(lines):
        key = (event.session_id, event.game_index)
        state = states.get(key)
        if state is None:
            state = _GameState(session_id=event.session_id, game_index=event.game_index)
            states[key] = state
            order.append(key)
        state.apply(event)
    games = tuple(states[key].finish() for key in order)
    logger.info(
        "Reconstructed %s game(s) and %s decision(s) from a runtime log",
        len(games),
        sum(len(game.decisions) for game in games),
    )
    return games


def decompose_played_log(
    path: Path,
    model: ModelRunnerConfig,
    *,
    run_root: Path | None = None,
) -> DecisionDecomposition:
    """Decompose the decisions a logged session made, re-scoring each one.

    The checkpoint has to be supplied, because a log identifies the settings a
    decision was made under but not the parameters behind it. Pointing this at a
    different checkpoint than the one that played is legitimate and sometimes
    the interesting question — what the current model would have thought of the
    decisions an older one made — so it is the caller's declaration rather than
    something inferred.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReconstructionError(f"cannot read {path}: {error}") from error
    decisions = tuple(
        decision for game in reconstruct_uci_games(lines) for decision in game.decisions
    )
    if not decisions:
        raise ReconstructionError(
            f"{path} records no engine decisions; a session logged below DEBUG "
            "emits no game events"
        )
    try:
        runner = CheckpointModelRunner.load(model, run_root=run_root)
    except ModelRunnerError as error:
        raise DecisionDecompositionError(
            f"cannot load the checkpoint to re-score with: {error}"
        ) from error
    return summarize_decisions(score_played_decisions(runner, decisions))


@dataclass(frozen=True)
class _Event:
    """One parsed game event."""

    session_id: str
    game_index: int
    event: str
    payload: dict[str, Any]


def _events(lines: Iterable[str]) -> tuple[_Event, ...]:
    """Parse the game events out of arbitrary log lines."""

    events: list[_Event] = []
    for number, line in enumerate(lines, start=1):
        marker = line.find(UCI_GAME_EVENT_PREFIX)
        if marker < 0:
            continue
        text = line[marker + len(UCI_GAME_EVENT_PREFIX) :].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ReconstructionError(
                f"line {number} carries an unreadable game event: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ReconstructionError(f"line {number} game event is not an object")
        schema = payload.get("schema")
        if schema != UCI_GAME_EVENT_SCHEMA:
            raise ReconstructionError(
                f"line {number} game event uses schema {schema!r}; this build "
                f"reads {UCI_GAME_EVENT_SCHEMA}"
            )
        events.append(
            _Event(
                session_id=_string(payload, "session_id", number),
                game_index=_integer(payload, "game_index", number),
                event=_string(payload, "event", number),
                payload=payload,
            )
        )
    return tuple(events)


class _GameState:
    """The engine state one game's events walk through.

    Mirrors what the adapter itself holds: a root position, the accepted move
    history, and nothing else. A decision is recorded against the state in force
    at that moment and then appended to it, which is exactly what the adapter
    does with the move it just chose.
    """

    def __init__(self, *, session_id: str, game_index: int) -> None:
        self._session_id = session_id
        self._game_index = game_index
        self._root_fen = chess.STARTING_FEN
        self._moves: tuple[chess.Move, ...] = ()
        self._decisions: list[PlayedDecision] = []

    def apply(self, event: _Event) -> None:
        """Fold one event into the reconstructed state."""

        if event.event == "position":
            self._accept_position(event)
        elif event.event == "decision":
            self._accept_decision(event)

    def finish(self) -> ReconstructedGame:
        """Return the reconstructed game."""

        return ReconstructedGame(
            session_id=self._session_id,
            game_index=self._game_index,
            initial_fen=self._root_fen,
            moves=self._moves,
            decisions=tuple(self._decisions),
        )

    def _accept_position(self, event: _Event) -> None:
        initial_fen = _string(event.payload, "initial_fen", None)
        moves = tuple(_move(value) for value in _strings(event.payload, "moves", None))
        board = _replay(initial_fen, moves)
        self._verify_position(event, board)
        self._root_fen = initial_fen
        self._moves = moves

    def _accept_decision(self, event: _Event) -> None:
        board = _replay(self._root_fen, self._moves)
        self._verify_position(event, board)
        observed = event.payload.get("observed_plies")
        if observed is not None and observed != len(self._moves):
            raise ReconstructionError(
                f"a decision in game {self._game_index} reports {observed} "
                f"observed plies where the log reconstructs {len(self._moves)}"
            )
        action_id = _integer(event.payload, "action_id", None)
        move = _move(_string(event.payload, "move", None))
        if move not in board.legal_moves:
            raise ReconstructionError(
                f"a decision in game {self._game_index} played {move.uci()}, "
                "which is illegal in the position the log reconstructs"
            )
        self._decisions.append(
            PlayedDecision(
                game_id=_game_id(self._session_id, self._game_index),
                ply_index=len(self._moves),
                initial_fen=self._root_fen,
                history=self._moves,
                action_id=action_id,
                config=_runtime_config(event.payload),
            )
        )
        self._moves = (*self._moves, move)

    def _verify_position(self, event: _Event, board: chess.Board) -> None:
        """Reject a log whose replay disagrees with the position it recorded.

        The adapter records the resulting position of every event it emits, so
        this is a free check that the reconstruction is the game that was
        played rather than a plausible one. An interleaved or truncated log
        fails here instead of producing quietly wrong decisions.
        """

        recorded = event.payload.get("position_fen")
        if recorded is not None and recorded != board.fen():
            raise ReconstructionError(
                f"a {event.event} event in game {self._game_index} records "
                f"position {recorded!r}, but its history replays to "
                f"{board.fen()!r}"
            )


def _runtime_config(payload: dict[str, Any]) -> RuntimeConfig:
    """Return the settings a logged decision was made under.

    The resolved seed is carried rather than the configured one, because that
    is the value that reproduces the session. It has no effect on the scored
    quantities, which come from the untempered policy, but a reader comparing a
    re-scored decision against the original game needs it.
    """

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ReconstructionError("a decision event carries no runtime settings")
    try:
        return RuntimeConfig(
            target_rating=runtime.get("target_rating"),
            temperature=runtime.get("temperature", 1.0),
            seed=runtime.get("resolved_seed"),
            resignation_enabled=bool(runtime.get("resignation_enabled", False)),
            draw_claim_enabled=bool(runtime.get("draw_claim_enabled", False)),
        )
    except ValueError as error:
        raise ReconstructionError(
            f"a decision event records unusable runtime settings: {error}"
        ) from error


def _replay(initial_fen: str, moves: tuple[chess.Move, ...]) -> chess.Board:
    try:
        board = chess.Board(initial_fen)
    except ValueError as error:
        raise ReconstructionError(f"invalid logged position: {error}") from error
    for ply_index, move in enumerate(moves):
        if move not in board.legal_moves:
            raise ReconstructionError(
                f"logged history is illegal at ply {ply_index}: {move.uci()}"
            )
        board.push(move)
    return board


def _move(value: str) -> chess.Move:
    try:
        return chess.Move.from_uci(value)
    except ValueError as error:
        raise ReconstructionError(f"invalid logged move {value!r}: {error}") from error


def _game_id(session_id: str, game_index: int) -> str:
    return f"{session_id}:{game_index}"


def _string(payload: dict[str, Any], key: str, line: int | None) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ReconstructionError(_missing(key, line))
    return value


def _strings(payload: dict[str, Any], key: str, line: int | None) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ReconstructionError(_missing(key, line))
    return tuple(value)


def _integer(payload: dict[str, Any], key: str, line: int | None) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ReconstructionError(_missing(key, line))
    return value


def _missing(key: str, line: int | None) -> str:
    where = "a game event" if line is None else f"the game event on line {line}"
    return f"{where} is missing a usable {key!r}"


__all__ = [
    "ReconstructedGame",
    "ReconstructionError",
    "decompose_played_log",
    "reconstruct_uci_games",
]
