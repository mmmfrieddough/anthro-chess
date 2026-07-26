"""One capture of code and environment provenance, shared by every producer."""

from __future__ import annotations

import re

import pytest

from anthro_chess import __version__
from anthro_chess.provenance import (
    code_provenance,
    environment_provenance,
    git_revision,
    optional_distribution_version,
)


def test_code_provenance_reports_this_build() -> None:
    code = code_provenance()

    assert code.package_version == __version__
    assert code.git_revision is None or re.fullmatch(r"[0-9a-f]{40}", code.git_revision)
    assert code.as_record() == {
        "package_version": __version__,
        "git_revision": code.git_revision,
    }


def test_environment_provenance_describes_the_machine() -> None:
    environment = environment_provenance()
    record = environment.as_record()

    assert environment.python_version
    assert environment.platform
    assert "torch" in environment.dependencies
    assert record["dependencies"] == dict(sorted(environment.dependencies.items()))


def test_a_missing_distribution_reports_none_instead_of_raising() -> None:
    assert optional_distribution_version("not-an-installed-distribution") is None


def test_git_revision_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance capture must not fail outside a checkout."""

    monkeypatch.setenv("PATH", "")

    assert git_revision() is None
