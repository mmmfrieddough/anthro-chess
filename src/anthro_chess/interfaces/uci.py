"""Minimal direct Universal Chess Interface adapter."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import chess
from pydantic import ValidationError

from anthro_chess import __version__
from anthro_chess.application_logging import (
    APPLICATION_LOGGER_NAME,
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_NAMES,
    configure_application_logging,
    default_uci_log_path,
)
from anthro_chess.config import ConfigError, load_config
from anthro_chess.inference import CheckpointModelRunner
from anthro_chess.interfaces.config import (
    UCI_MAX_RATING,
    UCI_MAX_SEED,
    UCI_MAX_TEMPERATURE,
    UCI_MIN_RATING,
    UCI_RANDOM_SEED,
    UCI_TEMPERATURE_SCALE,
    UciConfig,
)
from anthro_chess.machine import RUN_ROOT_VARIABLE, optional_root
from anthro_chess.runtime import (
    ActionModelRunner,
    DecisionRuntimeError,
    GameSession,
    MoveAction,
    RuntimeConfig,
)

ENGINE_NAME = "Anthro Chess"
ENGINE_AUTHOR = "Anthro Chess contributors"
NULL_BESTMOVE = "0000"
#: Identity and log-line marker of the DEBUG-tier game events this module
#: emits. This module owns the format; `anthro_chess.evaluation.reconstruction`
#: reads it back, so both sides share these constants rather than agreeing on a
#: string twice.
UCI_GAME_EVENT_SCHEMA = "anthro-uci-game-event-v1"
UCI_GAME_EVENT_PREFIX = "UCI game event "
_COMMANDS = frozenset(
    {
        "debug",
        "go",
        "isready",
        "ponderhit",
        "position",
        "quit",
        "setoption",
        "stop",
        "uci",
        "ucinewgame",
    }
)
RunnerLoader = Callable[[], ActionModelRunner]
logger = logging.getLogger(__name__)


class UciProtocolError(ValueError):
    """Raised when a UCI command cannot be applied safely."""


class UciInitializationError(RuntimeError):
    """Raised when the UCI process cannot initialize its model runtime."""


class UciInferenceError(RuntimeError):
    """Raised when an initialized model cannot produce a move."""


class UciEngine:
    """Translate one line-oriented UCI process into runtime calls."""

    def __init__(
        self,
        load_runner: RunnerLoader,
        config: UciConfig,
        *,
        normal_log_level: int = logging.INFO,
    ) -> None:
        self._load_runner = load_runner
        self._uci_elo = config.runtime.target_rating
        assert self._uci_elo is not None
        self._limit_strength = False
        self._runtime_config = RuntimeConfig.model_validate(
            config.runtime.model_copy(
                update={"target_rating": UCI_MAX_RATING}
            ).model_dump()
        )
        self._debug = False
        self._normal_log_level = normal_log_level
        self._log_session_id = uuid.uuid4().hex
        self._game_index = 0
        self._initial_fen = chess.STARTING_FEN
        self._moves: tuple[chess.Move, ...] = ()
        self._board = chess.Board()
        self._session: GameSession | None = None
        # ``go infinite`` must not answer until ``stop``, so the synchronously
        # computed response waits here instead of being sent immediately.
        self._pending_bestmove: str | None = None

    @property
    def board(self) -> chess.Board:
        """Return the current protocol position as a defensive copy."""

        return self._board.copy(stack=True)

    @property
    def runtime_config(self) -> RuntimeConfig:
        """Return the currently applied immutable runtime settings."""

        return self._runtime_config

    def run(
        self, input_stream: TextIO, output_stream: TextIO, error_stream: TextIO
    ) -> int:
        """Serve commands until EOF or ``quit``."""

        logger.info("UCI command loop started")
        for raw_line in input_stream:
            line = raw_line.strip()
            if not line:
                continue
            parsed = _find_command(line)
            if parsed is None:
                if self._debug:
                    logger.debug("Ignored unrecognized UCI input")
                continue
            command, arguments = parsed
            logger.debug("Received UCI command %s", command.lower())
            try:
                should_quit = self._handle(
                    command.lower(),
                    arguments.strip(),
                    output_stream,
                    error_stream,
                )
            except UciInitializationError as error:
                logger.error(
                    "%s",
                    error,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                return 2
            except UciInferenceError as error:
                self._send(
                    output_stream,
                    "info string CRITICAL ERROR: move generation failed",
                )
                logger.error(
                    "%s",
                    error,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                return 1
            except (UciProtocolError, DecisionRuntimeError, ValidationError) as error:
                logger.warning("%s", error)
                continue
            if should_quit:
                logger.info("UCI command loop stopped")
                return 0
        logger.info("UCI command loop reached end of input")
        return 0

    def _handle(
        self,
        command: str,
        arguments: str,
        output: TextIO,
        error: TextIO,
    ) -> bool:
        if command == "uci":
            self._identify(output)
        elif command == "isready":
            self._ensure_initialized(error)
            self._send(output, "readyok")
        elif command == "debug":
            self._set_debug(arguments)
        elif command == "setoption":
            self._set_option(arguments)
        elif command == "ucinewgame":
            self._new_game()
        elif command == "position":
            initial_fen, moves = _parse_position(arguments)
            self._sync_position(initial_fen, moves)
        elif command == "go":
            self._go(arguments, output, error)
        elif command == "stop":
            self._stop(output)
        elif command == "ponderhit":
            pass
        elif command == "quit":
            return True
        elif self._debug:
            self._send(output, f"info string ignored command: {command}")
        return False

    def _identify(self, output: TextIO) -> None:
        temperature = round(self._runtime_config.temperature * UCI_TEMPERATURE_SCALE)
        self._send(output, f"id name {ENGINE_NAME} {__version__}")
        self._send(output, f"id author {ENGINE_AUTHOR}")
        self._send(
            output,
            "option name UCI_LimitStrength type check default false",
        )
        self._send(
            output,
            "option name UCI_Elo type spin "
            f"default {self._uci_elo} min {UCI_MIN_RATING} max {UCI_MAX_RATING}",
        )
        self._send(
            output,
            "option name Anthro Temperature type spin "
            f"default {temperature} "
            f"min 0 max {UCI_MAX_TEMPERATURE}",
        )
        seed = self._runtime_config.seed
        seed_default = UCI_RANDOM_SEED if seed is None else seed
        self._send(
            output,
            "option name Anthro Seed type spin "
            f"default {seed_default} "
            f"min {UCI_RANDOM_SEED} max {UCI_MAX_SEED}",
        )
        self._send(output, "uciok")

    def _set_debug(self, arguments: str) -> None:
        normalized = arguments.lower()
        if normalized not in {"on", "off"}:
            raise UciProtocolError("debug expects 'on' or 'off'")
        self._debug = normalized == "on"
        application_logger = logging.getLogger(APPLICATION_LOGGER_NAME)
        application_logger.setLevel(
            logging.DEBUG if self._debug else self._normal_log_level
        )
        logger.debug("UCI debug logging enabled")
        self._log_game_event("session", runtime=self._runtime_log_record())

    def _set_option(self, arguments: str) -> None:
        name, value = _parse_option(arguments)
        # Engine options are bounded scalars a GUI chose deliberately, and the
        # resolved seed is already logged in full. Recording what a GUI asked
        # for is what makes a reported session reproducible.
        logger.debug("Received UCI option %s with value %s", name, value)
        normalized = name.casefold()
        if normalized == "uci_limitstrength":
            limit_strength = _parse_check(value, name=name)
            self._limit_strength = limit_strength
            target_rating = self._uci_elo if limit_strength else UCI_MAX_RATING
            self._replace_runtime(target_rating=target_rating)
        elif normalized == "uci_elo":
            rating = _parse_spin(
                value,
                name=name,
                minimum=UCI_MIN_RATING,
                maximum=UCI_MAX_RATING,
            )
            self._uci_elo = rating
            if self._limit_strength:
                self._replace_runtime(target_rating=rating)
        elif normalized == "anthro temperature":
            temperature = _parse_spin(
                value,
                name=name,
                minimum=0,
                maximum=UCI_MAX_TEMPERATURE,
            )
            self._replace_runtime(
                temperature=temperature / UCI_TEMPERATURE_SCALE,
            )
        elif normalized == "anthro seed":
            seed = _parse_spin(
                value,
                name=name,
                minimum=UCI_RANDOM_SEED,
                maximum=UCI_MAX_SEED,
            )
            self._set_seed(None if seed == UCI_RANDOM_SEED else seed)
        else:
            return

    def _replace_runtime(self, **update: object) -> None:
        replacement = self._runtime_config.model_copy(update=update)
        replacement = RuntimeConfig.model_validate(replacement.model_dump())
        self._runtime_config = replacement
        if self._session is not None:
            self._session.config = replacement

    def _set_seed(self, seed: int | None) -> None:
        self._replace_runtime(seed=seed)
        if self._session is not None:
            self._session.reseed()
        logger.debug("Applied seed policy: %s", "random" if seed is None else seed)

    def _new_game(self) -> None:
        self._discard_pending()
        self._game_index += 1
        self._initial_fen = chess.STARTING_FEN
        self._moves = ()
        self._board = chess.Board()
        if self._session is not None:
            self._session.reset()
        self._log_game_event("new-game", runtime=self._runtime_log_record())
        logger.debug("Started a new game")

    def _sync_position(
        self,
        initial_fen: str,
        moves: tuple[chess.Move, ...],
    ) -> None:
        sync = None
        if self._session is not None:
            sync = self._session.sync_position(initial_fen=initial_fen, moves=moves)
            board = self._session.board
        else:
            board = _validated_board(initial_fen, moves)
        self._initial_fen = initial_fen
        self._moves = moves
        self._board = board
        self._log_game_event(
            "position",
            initial_fen=initial_fen,
            moves=[move.uci() for move in moves],
            position_fen=board.fen(),
            transition=(
                "validated"
                if sync is None
                else "replaced"
                if sync.replaced
                else "append-only"
            ),
        )
        logger.debug("Synced UCI position with %s observed plies", len(moves))

    def _ensure_initialized(self, error_stream: TextIO) -> GameSession:
        if self._session is not None:
            return self._session
        try:
            with contextlib.redirect_stdout(error_stream):
                runner = self._load_runner()
            session = GameSession(
                runner,
                config=self._runtime_config,
                initial_fen=self._initial_fen,
                moves=self._moves,
            )
        except Exception as error:
            raise UciInitializationError(
                f"model initialization failed: {error}"
            ) from error
        self._session = session
        self._log_game_event("runtime", runtime=self._runtime_log_record())
        logger.info("Model runtime initialized")
        return session

    def _go(self, arguments: str, output: TextIO, error_stream: TextIO) -> None:
        session = self._ensure_initialized(error_stream)
        infinite = "infinite" in {token.lower() for token in arguments.split()}
        if session.is_terminal:
            self._respond(NULL_BESTMOVE, output, infinite=infinite)
            return
        position_fen = session.board.fen()
        try:
            action = session.choose_action()
        except Exception as error:
            raise UciInferenceError(f"move generation failed: {error}") from error
        if not isinstance(action, MoveAction):
            raise UciProtocolError(
                "portable UCI mode cannot represent the terminal action "
                f"{action.action_id}"
            )
        self._log_game_event(
            "decision",
            action_id=action.action_id,
            move=action.move.uci(),
            observed_plies=len(self._moves),
            position_fen=position_fen,
            runtime=self._runtime_log_record(),
        )
        self._moves += (action.move,)
        self._board = session.board
        self._respond(action.move.uci(), output, infinite=infinite)

    def _respond(self, bestmove: str, output: TextIO, *, infinite: bool) -> None:
        if infinite:
            # Search is synchronous, so the answer already exists; the protocol
            # just forbids sending it before the GUI asks to stop.
            self._pending_bestmove = bestmove
            logger.debug("Holding an infinite-search response until stop")
            return
        self._send(output, f"bestmove {bestmove}")

    def _stop(self, output: TextIO) -> None:
        pending, self._pending_bestmove = self._pending_bestmove, None
        if pending is None:
            # An unpaired bestmove is a worse violation than ignoring stop.
            return
        self._send(output, f"bestmove {pending}")

    def _discard_pending(self) -> None:
        if self._pending_bestmove is not None:
            self._pending_bestmove = None
            logger.debug("Discarded an unstopped infinite-search response")

    def _runtime_log_record(self) -> dict[str, bool | float | int | None]:
        session = self._session
        return {
            "target_rating": self._runtime_config.target_rating,
            "temperature": self._runtime_config.temperature,
            "resignation_enabled": self._runtime_config.resignation_enabled,
            "draw_claim_enabled": self._runtime_config.draw_claim_enabled,
            "configured_seed": self._runtime_config.seed,
            "resolved_seed": None if session is None else session.resolved_seed,
        }

    def _log_game_event(self, event: str, **fields: object) -> None:
        payload = {
            "schema": UCI_GAME_EVENT_SCHEMA,
            "session_id": self._log_session_id,
            "game_index": self._game_index,
            "event": event,
            **fields,
        }
        logger.debug(
            "%s%s",
            UCI_GAME_EVENT_PREFIX,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _send(output: TextIO, message: str) -> None:
        print(message, file=output, flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the dedicated UCI process parser."""

    parser = argparse.ArgumentParser(
        prog="anthro-uci",
        description="Run Anthro Chess as a directly invoked UCI engine.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional strict TOML model and runtime configuration.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted TOML override; may be repeated.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVEL_NAMES,
        default=DEFAULT_LOG_LEVEL,
        help="UCI file log verbosity (default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Rotating UCI log file (default: platform-local application logs).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load strict configuration and serve UCI on standard streams."""

    arguments = build_parser().parse_args(argv)
    logging_configuration = configure_application_logging(
        level=arguments.log_level,
        file_path=arguments.log_file or default_uci_log_path(),
        stream=sys.stderr,
    )
    if logging_configuration.file_path is not None:
        logger.info("UCI logs initialized at %s", logging_configuration.file_path)
    try:
        resolved = load_config(
            UciConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        run_root = optional_root(RUN_ROOT_VARIABLE)
    except ConfigError as error:
        logger.error("Configuration failed: %s", error)
        return 2

    def load_runner() -> ActionModelRunner:
        return CheckpointModelRunner.load(
            resolved.value.model,
            run_root=run_root,
        )

    return UciEngine(
        load_runner,
        resolved.value,
        normal_log_level=logging_configuration.level,
    ).run(
        sys.stdin,
        sys.stdout,
        sys.stderr,
    )


def _find_command(line: str) -> tuple[str, str] | None:
    tokens = line.split()
    for index, token in enumerate(tokens):
        command = token.lower()
        if command in _COMMANDS:
            return command, " ".join(tokens[index + 1 :])
    return None


def _parse_position(arguments: str) -> tuple[str, tuple[chess.Move, ...]]:
    tokens = arguments.split()
    if not tokens:
        raise UciProtocolError("position requires 'startpos' or 'fen'")

    if tokens[0].lower() == "startpos":
        initial_fen = chess.STARTING_FEN
        remainder = tokens[1:]
    elif tokens[0].lower() == "fen":
        if len(tokens) < 7:
            raise UciProtocolError("position fen requires all six FEN fields")
        initial_fen = " ".join(tokens[1:7])
        remainder = tokens[7:]
    else:
        raise UciProtocolError("position requires 'startpos' or 'fen'")

    move_marker = next(
        (index for index, token in enumerate(remainder) if token.lower() == "moves"),
        None,
    )
    move_tokens = remainder[move_marker + 1 :] if move_marker is not None else []
    moves: list[chess.Move] = []
    for ply_index, token in enumerate(move_tokens):
        try:
            moves.append(chess.Move.from_uci(token))
        except ValueError as error:
            # Skipping the token would shift every later move one ply earlier
            # and silently desynchronize the engine from the GUI, so reject the
            # whole command and keep the previously synchronized position.
            raise UciProtocolError(
                f"malformed position move at ply {ply_index}: {token}"
            ) from error
    return initial_fen, tuple(moves)


def _parse_option(arguments: str) -> tuple[str, str]:
    tokens = arguments.split()
    try:
        name_index = next(
            index for index, token in enumerate(tokens) if token.lower() == "name"
        )
    except StopIteration as error:
        raise UciProtocolError("setoption requires 'name'") from error
    option_tokens = tokens[name_index:]
    try:
        value_index = next(
            index
            for index, token in enumerate(option_tokens[1:], start=1)
            if token.lower() == "value"
        )
    except StopIteration as error:
        raise UciProtocolError("setoption requires 'value'") from error
    name = " ".join(option_tokens[1:value_index]).strip()
    value = " ".join(option_tokens[value_index + 1 :]).strip()
    if not name or not value:
        raise UciProtocolError("setoption requires nonempty name and value")
    return name, value


def _parse_check(value: str, *, name: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise UciProtocolError(f"{name} expects true or false")
    return normalized == "true"


def _parse_spin(value: str, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise UciProtocolError(f"{name} expects an integer") from error
    if not minimum <= parsed <= maximum:
        raise UciProtocolError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _validated_board(
    initial_fen: str,
    moves: tuple[chess.Move, ...],
) -> chess.Board:
    try:
        board = chess.Board(initial_fen)
    except ValueError as error:
        raise UciProtocolError(f"invalid position FEN: {error}") from error
    for ply_index, move in enumerate(moves):
        if move not in board.legal_moves:
            raise UciProtocolError(
                f"illegal position move at ply {ply_index}: {move.uci()}"
            )
        board.push(move)
    return board


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
