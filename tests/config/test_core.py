from pathlib import Path

import pytest
from pydantic import StrictBool

from anthro_chess.config import ConfigError, ConfigModel, load_config


class RuntimeSettings(ConfigModel):
    temperature: float = 1.0
    enabled: StrictBool = False


class ExampleConfig(ConfigModel):
    name: str = "default"
    output: Path | None = None
    runtime: RuntimeSettings = RuntimeSettings()


def test_loads_code_owned_defaults_without_a_file() -> None:
    resolved = load_config(ExampleConfig)

    assert resolved.value == ExampleConfig()
    assert resolved.provenance.source is None
    assert resolved.provenance.overrides == ()


def test_loads_toml_and_applies_strict_dotted_overrides(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        'name = "sample"\noutput = "artifacts/run"\n\n[runtime]\ntemperature = 0.8\n',
        encoding="utf-8",
    )

    resolved = load_config(
        ExampleConfig,
        path=path,
        overrides=("runtime.temperature=0.65", "runtime.enabled=true"),
    )

    assert resolved.value.name == "sample"
    assert resolved.value.output == Path("artifacts/run")
    assert resolved.value.runtime == RuntimeSettings(temperature=0.65, enabled=True)
    assert resolved.provenance.source == str(path.resolve())
    assert resolved.as_record() == {
        "config": {
            "name": "sample",
            "output": "artifacts/run",
            "runtime": {"temperature": 0.65, "enabled": True},
        },
        "provenance": {
            "source": str(path.resolve()),
            "overrides": ["runtime.temperature=0.65", "runtime.enabled=true"],
        },
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("unknown = 1\n", "Extra inputs are not permitted"),
        ('[runtime]\nextra = "no"\n', "Extra inputs are not permitted"),
        ('[runtime]\nenabled = "yes"\n', "Input should be a valid boolean"),
    ],
)
def test_rejects_unknown_or_invalid_values(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(ExampleConfig, path=path)


def test_rejects_invalid_override_syntax() -> None:
    with pytest.raises(ConfigError, match="expected dotted.key"):
        load_config(ExampleConfig, overrides=("runtime.temperature",))
