"""Strict loading for code-owned configuration schemas."""

from __future__ import annotations

import dataclasses
import tomllib
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

ConfigT = TypeVar("ConfigT")


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
        """Return a serializable record suitable for run or artifact metadata."""
        return cast(
            dict[str, object],
            _record_value(
                {
                    "config": dataclasses.asdict(cast(Any, self.value)),
                    "provenance": dataclasses.asdict(self.provenance),
                }
            ),
        )


def load_config(
    schema: type[ConfigT],
    *,
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> ResolvedConfig[ConfigT]:
    """Load defaults, an optional TOML selection, and dotted TOML overrides.

    ``schema`` must be a dataclass whose fields have defaults. The schema class is
    the owner of field names, types, validation, and default values; checked-in
    TOML files merely select values. Relative paths are interpreted by the caller
    in the normal operating-system way and no repository directory is searched.
    """
    if not dataclasses.is_dataclass(schema):
        raise TypeError("configuration schema must be a dataclass type")

    try:
        defaults = schema()
    except TypeError as error:
        raise TypeError("configuration schema fields must have defaults") from error

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

    value = _decode_dataclass(schema, raw, defaults, path="config")
    provenance = ConfigProvenance(
        source=str(source.resolve()) if source is not None else None,
        overrides=override_items,
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


def _decode_dataclass(
    schema: type[ConfigT],
    supplied: Mapping[str, object],
    defaults: ConfigT,
    *,
    path: str,
) -> ConfigT:
    fields = {field.name: field for field in dataclasses.fields(cast(Any, schema))}
    unknown = sorted(set(supplied) - set(fields))
    if unknown:
        names = ", ".join(f"{path}.{name}" for name in unknown)
        raise ConfigError(f"unknown configuration field(s): {names}")

    hints = get_type_hints(schema)
    values: dict[str, object] = {}
    for name, field in fields.items():
        default = getattr(defaults, name)
        if name not in supplied:
            values[name] = default
            continue
        values[name] = _decode_value(
            supplied[name], hints.get(name, field.type), default, path=f"{path}.{name}"
        )

    try:
        return schema(**values)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid {path}: {error}") from error


def _decode_value(
    value: object, annotation: object, default: object, *, path: str
) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a table")
        return _decode_dataclass(cast(type[Any], annotation), value, default, path=path)

    if origin in (Union, types.UnionType):
        failures: list[str] = []
        for option in args:
            if option is type(None) and value is None:
                return None
            try:
                return _decode_value(value, option, default, path=path)
            except ConfigError as error:
                failures.append(str(error))
        raise ConfigError(
            f"{path} does not match any allowed type: {'; '.join(failures)}"
        )

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be an array")
        item_type = args[0] if args else Any
        return [
            _decode_value(item, item_type, None, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin is dict:
        key_type, item_type = args if args else (Any, Any)
        if key_type not in (str, Any) or not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a string-keyed table")
        return {
            key: _decode_value(item, item_type, None, path=f"{path}.{key}")
            for key, item in value.items()
        }

    if annotation is Any:
        return value
    if (
        annotation is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)
    if annotation in (bool, int, str):
        if type(value) is not annotation:
            raise ConfigError(f"{path} must be {annotation.__name__}")
        return value
    if annotation is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a path string")
        return Path(value).expanduser()

    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    raise ConfigError(f"{path} uses unsupported configuration type {annotation!r}")


def _record_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _record_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record_value(item) for item in value]
    return value
