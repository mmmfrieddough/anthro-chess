"""Process-level logging configuration for Anthro Chess applications."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

APPLICATION_LOGGER_NAME = "anthro_chess"
LOG_LEVEL_NAMES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_BACKUP_COUNT = 3
_OWNED_HANDLER_ATTRIBUTE = "_anthro_chess_application_handler"


@dataclass(frozen=True)
class LoggingConfiguration:
    """Resolved destination and level for one process logging setup."""

    level: int
    file_path: Path | None
    using_fallback: bool


def configure_application_logging(
    *,
    level: str | int = DEFAULT_LOG_LEVEL,
    stream: TextIO | None = None,
    file_path: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> LoggingConfiguration:
    """Configure the application logger for a console or rotating file."""

    resolved_level = parse_log_level(level)
    application_logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    application_logger.setLevel(resolved_level)
    application_logger.propagate = False
    _remove_owned_handlers(application_logger)

    resolved_path = (
        Path(file_path).expanduser().resolve() if file_path is not None else None
    )
    using_fallback = False
    if resolved_path is None:
        handler: logging.Handler = logging.StreamHandler(stream or sys.stderr)
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    else:
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                resolved_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        except OSError as error:
            using_fallback = True
            resolved_path = None
            handler = logging.StreamHandler(stream or sys.stderr)
            formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(logging.NOTSET)
            setattr(handler, _OWNED_HANDLER_ATTRIBUTE, True)
            application_logger.addHandler(handler)
            application_logger.warning(
                "Could not initialize rotating file logging; using standard error: %s",
                error,
            )
            return LoggingConfiguration(
                level=resolved_level,
                file_path=None,
                using_fallback=True,
            )

    handler.setFormatter(formatter)
    handler.setLevel(logging.NOTSET)
    setattr(handler, _OWNED_HANDLER_ATTRIBUTE, True)
    application_logger.addHandler(handler)
    return LoggingConfiguration(
        level=resolved_level,
        file_path=resolved_path,
        using_fallback=using_fallback,
    )


def parse_log_level(level: str | int) -> int:
    """Return a supported standard logging level."""

    if isinstance(level, int):
        if level in {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }:
            return level
        raise ValueError(f"unsupported logging level: {level}")
    normalized = level.upper()
    if normalized not in LOG_LEVEL_NAMES:
        raise ValueError(f"unsupported logging level: {level}")
    return int(getattr(logging, normalized))


def default_uci_log_path() -> Path:
    """Return the code-owned default location for bounded UCI process logs."""

    configured_root = os.environ.get("ANTHRO_CHESS_LOG_ROOT", "").strip()
    if configured_root:
        root = Path(configured_root).expanduser()
    elif os.environ.get("XDG_STATE_HOME", "").strip():
        root = Path(os.environ["XDG_STATE_HOME"]).expanduser() / "anthro-chess"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Logs" / "Anthro Chess"
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA", "").strip():
        root = Path(os.environ["LOCALAPPDATA"]).expanduser() / "Anthro Chess" / "Logs"
    else:
        root = Path.home() / ".local" / "state" / "anthro-chess"
    return root / "uci.log"


def _remove_owned_handlers(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()
