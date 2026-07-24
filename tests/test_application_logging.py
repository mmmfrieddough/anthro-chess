from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pytest

from anthro_chess.application_logging import (
    APPLICATION_LOGGER_NAME,
    configure_application_logging,
    default_uci_log_path,
)


def test_console_logging_filters_levels_and_repeated_setup_does_not_duplicate() -> None:
    first = StringIO()
    configure_application_logging(level="WARNING", stream=first)
    logger = logging.getLogger(f"{APPLICATION_LOGGER_NAME}.test")
    logger.info("hidden")
    logger.warning("first")

    second = StringIO()
    configure_application_logging(level="DEBUG", stream=second)
    logger.debug("second")

    assert first.getvalue().count("first") == 1
    assert "hidden" not in first.getvalue()
    assert second.getvalue().count("second") == 1
    assert "first" not in second.getvalue()


def test_rotating_file_logging_is_bounded(tmp_path: Path) -> None:
    log_path = tmp_path / "logs/uci.log"
    configuration = configure_application_logging(
        level="INFO",
        file_path=log_path,
        max_bytes=120,
        backup_count=1,
    )
    logger = logging.getLogger(f"{APPLICATION_LOGGER_NAME}.test")

    for index in range(20):
        logger.info("bounded record %s with enough content to rotate", index)

    assert configuration.file_path == log_path.resolve()
    assert not configuration.using_fallback
    assert log_path.is_file()
    assert log_path.with_name("uci.log.1").is_file()
    assert len(tuple(log_path.parent.glob("uci.log*"))) == 2


def test_file_setup_failure_falls_back_to_standard_error(tmp_path: Path) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")
    fallback = StringIO()

    configuration = configure_application_logging(
        level="INFO",
        file_path=blocking_file / "uci.log",
        stream=fallback,
    )
    logging.getLogger(f"{APPLICATION_LOGGER_NAME}.test").info("still available")

    assert configuration.file_path is None
    assert configuration.using_fallback
    assert "using standard error" in fallback.getvalue()
    assert "still available" in fallback.getvalue()


def test_default_uci_log_path_honors_application_log_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHRO_CHESS_LOG_ROOT", str(tmp_path))

    assert default_uci_log_path() == tmp_path / "uci.log"
