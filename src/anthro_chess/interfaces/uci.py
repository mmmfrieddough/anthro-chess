"""Minimal direct Universal Chess Interface adapter."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import chess
from pydantic import ValidationError

from anthro_chess import __version__
from anthro_chess.config import ConfigError, load_config
from anthro_chess.inference import CheckpointModelRunner, ModelRunnerError
from anthro_chess.interfaces.config import (
    UCI_MAX_RATING,
    UCI_MAX_TEMPERATURE,
    UCI_MIN_RATING,
    UCI_TEMPERATURE_SCALE,
    UciConfig,
)
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


class UciProtocolError(ValueError):
    """Raised when a UCI command cannot be applied safely."""


class UciEngine:
    """Translate one line-oriented UCI process into runtime calls."""

    def __init__(self, runner: ActionModelRunner, config: UciConfig) -> None:
        self._runner = runner
        self._runtime_config = config.runtime
        self._uci_elo = config.runtime.target_rating
        assert self._uci_elo is not None
        self._limit_strength = True
        self._debug = False
        self._initial_fen = chess.STARTING_FEN
        self._moves: tuple[chess.Move, ...] = ()
        self._session = self._build_session(self._initial_fen, self._moves)

    @property
    def board(self) -> chess.Board:
        """Return the current protocol position as a defensive copy."""

        return self._session.board

    @property
    def runtime_config(self) -> RuntimeConfig:
        """Return the currently applied immutable runtime settings."""

        return self._runtime_config

    def run(
        self, input_stream: TextIO, output_stream: TextIO, error_stream: TextIO
    ) -> int:
        """Serve commands until EOF or ``quit``."""

        for raw_line in input_stream:
            line = raw_line.strip()
            if not line:
                continue
            command, _, arguments = line.partition(" ")
            try:
                should_quit = self._handle(
                    command.lower(),
                    arguments.strip(),
                    output_stream,
                    error_stream,
                )
            except (UciProtocolError, DecisionRuntimeError, ValidationError) as error:
                print(f"anthro-uci: {error}", file=error_stream, flush=True)
                if self._debug:
                    self._send(output_stream, f"info string error: {error}")
                continue
            if should_quit:
                return 0
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
            self._send(output, "readyok")
        elif command == "debug":
            self._set_debug(arguments)
        elif command == "setoption":
            self._set_option(arguments)
        elif command == "ucinewgame":
            self._replace_position(chess.STARTING_FEN, ())
        elif command == "position":
            initial_fen, moves = _parse_position(arguments)
            self._replace_position(initial_fen, moves)
        elif command == "go":
            self._go(output, error)
        elif command in {"stop", "ponderhit"}:
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
            "option name UCI_LimitStrength type check default true",
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
        self._send(output, "uciok")

    def _set_debug(self, arguments: str) -> None:
        normalized = arguments.lower()
        if normalized not in {"on", "off"}:
            raise UciProtocolError("debug expects 'on' or 'off'")
        self._debug = normalized == "on"

    def _set_option(self, arguments: str) -> None:
        name, value = _parse_option(arguments)
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
        else:
            raise UciProtocolError(f"unsupported option: {name}")

    def _replace_runtime(
        self,
        *,
        target_rating: int | None = None,
        temperature: float | None = None,
    ) -> None:
        update: dict[str, object] = {}
        if target_rating is not None:
            update["target_rating"] = target_rating
        if temperature is not None:
            update["temperature"] = temperature
        replacement = self._runtime_config.model_copy(update=update)
        replacement = RuntimeConfig.model_validate(replacement.model_dump())
        self._runtime_config = replacement
        self._session.config = replacement

    def _replace_position(
        self,
        initial_fen: str,
        moves: tuple[chess.Move, ...],
    ) -> None:
        replacement = self._build_session(initial_fen, moves)
        self._initial_fen = initial_fen
        self._moves = moves
        self._session = replacement

    def _build_session(
        self,
        initial_fen: str,
        moves: tuple[chess.Move, ...],
    ) -> GameSession:
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
        session = GameSession(
            self._runner,
            controlled_color=board.turn,
            config=self._runtime_config,
        )
        session.reset(initial_fen=initial_fen, moves=moves)
        return session

    def _go(self, output: TextIO, error_stream: TextIO) -> None:
        if self._session.is_terminal:
            self._send(output, f"bestmove {NULL_BESTMOVE}")
            return
        try:
            action = self._session.choose_action()
        except Exception as error:
            print(
                f"anthro-uci: move generation failed: {error}",
                file=error_stream,
                flush=True,
            )
            self._send(output, f"bestmove {NULL_BESTMOVE}")
            return
        if not isinstance(action, MoveAction):
            raise UciProtocolError("portable UCI mode cannot represent resignation")
        self._moves += (action.move,)
        self._send(output, f"bestmove {action.move.uci()}")

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load the selected checkpoint and serve UCI on standard streams."""

    arguments = build_parser().parse_args(argv)
    try:
        resolved = load_config(
            UciConfig,
            path=arguments.config,
            overrides=arguments.set,
        )
        run_root_value = os.environ.get("ANTHRO_CHESS_RUN_ROOT", "").strip()
        run_root = (
            Path(run_root_value).expanduser().resolve() if run_root_value else None
        )
        runner = CheckpointModelRunner.load(
            resolved.value.model,
            run_root=run_root,
        )
    except (ConfigError, ModelRunnerError) as error:
        print(f"anthro-uci: {error}", file=sys.stderr)
        return 2

    return UciEngine(runner, resolved.value).run(sys.stdin, sys.stdout, sys.stderr)


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

    if remainder:
        if remainder[0].lower() != "moves":
            raise UciProtocolError("unexpected position fields after the base position")
        remainder = remainder[1:]

    try:
        moves = tuple(chess.Move.from_uci(token) for token in remainder)
    except ValueError as error:
        raise UciProtocolError(f"invalid UCI move in position: {error}") from error
    return initial_fen, moves


def _parse_option(arguments: str) -> tuple[str, str]:
    tokens = arguments.split()
    if not tokens or tokens[0].lower() != "name":
        raise UciProtocolError("setoption requires 'name'")
    try:
        value_index = next(
            index
            for index, token in enumerate(tokens[1:], start=1)
            if token.lower() == "value"
        )
    except StopIteration as error:
        raise UciProtocolError("setoption requires 'value'") from error
    name = " ".join(tokens[1:value_index]).strip()
    value = " ".join(tokens[value_index + 1 :]).strip()
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


if __name__ == "__main__":  # pragma: no cover - console scripts call main directly
    raise SystemExit(main())
