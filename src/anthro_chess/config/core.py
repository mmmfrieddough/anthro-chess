"""Strict loading for code-owned configuration schemas."""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class ConfigModel(BaseModel):
    """Base for immutable, code-owned command configuration schemas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


ConfigT = TypeVar("ConfigT", bound=ConfigModel)


class ConfigError(ValueError):
    """Raised when configuration input does not match its code-owned schema."""


@dataclass(frozen=True)
class ConfigProvenance:
    """Inputs needed to explain how a resolved configuration was selected."""

    source: str | None
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedConfig(Generic[ConfigT]):
    """A validated configuration plus its selection provenance."""

    value: ConfigT
    provenance: ConfigProvenance

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible record for run or artifact metadata."""
        return {
            "config": self.value.model_dump(mode="json"),
            "provenance": {
                "source": self.provenance.source,
                "overrides": list(self.provenance.overrides),
            },
        }


def load_config(
    schema: type[ConfigT],
    *,
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> ResolvedConfig[ConfigT]:
    """Load defaults, an optional TOML selection, and dotted TOML overrides."""
    if not issubclass(schema, ConfigModel):
        raise TypeError("configuration schema must inherit from ConfigModel")

    source = Path(path).expanduser() if path is not None else None
    raw: dict[str, object] = {}
    if source is not None:
        try:
            with source.open("rb") as config_file:
                raw = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError(f"cannot load configuration {source}: {error}") from error

    override_items = tuple(overrides)
    for override in override_items:
        _apply_override(raw, override)

    try:
        value = schema.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid configuration: {error}") from error

    provenance = ConfigProvenance(
        source=str(source.resolve()) if source is not None else None,
        overrides=override_items,
    )
    logger.debug(
        "Loaded %s configuration from %s with %s override(s)",
        schema.__name__,
        provenance.source or "defaults",
        len(override_items),
    )
    return ResolvedConfig(value=value, provenance=provenance)


def _apply_override(target: dict[str, object], override: str) -> None:
    key, separator, raw_value = override.partition("=")
    parts = key.split(".")
    if not separator or not raw_value or any(not part for part in parts):
        raise ConfigError(
            f"invalid override {override!r}; expected dotted.key=<TOML value>"
        )

    try:
        value = tomllib.loads(f"value = {raw_value}")["value"]
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"invalid TOML value in override {override!r}: {error}"
        ) from error

    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override path {key!r} crosses a non-table value")
        current = child
    current[parts[-1]] = value
