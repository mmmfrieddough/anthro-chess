"""Shared capture of the code and environment that produced an artifact.

Run records, checkpoints, and benchmark results all have to answer "what
produced this". Capturing that in one place keeps the answer identical across
producers instead of drifting into per-artifact reimplementations.

None of this belongs in a comparability fingerprint. Software versions and
platform details describe how a measurement was produced, not what it
measured, and decision 0013 deliberately keeps them out of series identity.

Efficiency metrics need some of the same facts as realized inputs rather than
as provenance, and they get them from an execution component that is captured
and digested separately. See
``docs/decisions/0018-execution-sensitive-efficiency-series.md``; this module
stays the provenance side of that split.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

from anthro_chess import __version__

#: Distributions whose versions are worth recording when they are installed.
OPTIONAL_DISTRIBUTIONS = ("torch", "pyarrow", "chess")


def git_revision() -> str | None:
    """Return the current Git revision, or ``None`` outside a checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def optional_distribution_version(name: str) -> str | None:
    """Return an installed distribution version without importing the package."""

    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class CodeProvenance:
    """Which build of this project produced an artifact."""

    package_version: str
    git_revision: str | None

    def as_record(self) -> dict[str, str | None]:
        """Return the stable record embedded in artifacts."""

        return {
            "package_version": self.package_version,
            "git_revision": self.git_revision,
        }


@dataclass(frozen=True)
class EnvironmentProvenance:
    """Where an artifact was produced, beyond this project's own code."""

    python_version: str
    platform: str
    dependencies: dict[str, str | None]

    def as_record(self) -> dict[str, object]:
        """Return the stable record embedded in artifacts."""

        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "dependencies": dict(sorted(self.dependencies.items())),
        }


def code_provenance() -> CodeProvenance:
    """Capture the package version and checkout revision."""

    return CodeProvenance(
        package_version=__version__,
        git_revision=git_revision(),
    )


def environment_provenance() -> EnvironmentProvenance:
    """Capture the interpreter, platform, and installed dependency versions."""

    return EnvironmentProvenance(
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependencies={
            name: optional_distribution_version(name) for name in OPTIONAL_DISTRIBUTIONS
        },
    )
