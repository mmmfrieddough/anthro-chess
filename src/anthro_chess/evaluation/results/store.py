"""The durable results store benchmarks append to and reports read from.

The store is layered. The committed summary tier holds one small JSON file per
result, so history is versioned with the code, metric movement appears as a
reviewable diff, and an agent reads results with ordinary file tools rather
than through a service. The machine-local detail tier holds per-position
diagnostics, slice tables, and generated games, and is referenced from the
summary rather than copied into it.

One file per result is what makes the committed tier safe to append to: two
sessions recording different results write different files, so a shared
history file cannot be corrupted or fought over. Writing the same result twice
is idempotent, and writing a different result to the same identity fails
loudly.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from anthro_chess.evaluation.results.noise import (
    NoiseCharacterization,
    NoiseCharacterizationError,
)
from anthro_chess.evaluation.results.records import (
    Bridge,
    DetailReference,
    ResultEnvelope,
    ResultRecordError,
)

RecordT = TypeVar("RecordT", bound=BaseModel)

#: Directory name of the committed summary tier, relative to the repository.
DEFAULT_STORE_DIRECTORY = "results"
STORE_ROOT_VARIABLE = "ANTHRO_CHESS_RESULTS_ROOT"
DETAIL_ROOT_VARIABLE = "ANTHRO_CHESS_RESULT_DETAIL_ROOT"

RECORDS_DIRECTORY = "records"
BRIDGES_DIRECTORY = "bridges"
FLOORS_DIRECTORY = "floors"
LOCK_FILE_NAME = ".write-lock"

logger = logging.getLogger(__name__)


class ResultsStoreError(ValueError):
    """Raised when the results store cannot be read or appended to safely."""


class ResultsStore:
    """Read and append the committed summary tier."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Return the store root."""

        return self._root

    @property
    def records_directory(self) -> Path:
        """Return the directory holding committed result records."""

        return self._root / RECORDS_DIRECTORY

    @property
    def bridges_directory(self) -> Path:
        """Return the directory holding committed bridges."""

        return self._root / BRIDGES_DIRECTORY

    @property
    def floors_directory(self) -> Path:
        """Return the directory holding committed noise characterizations."""

        return self._root / FLOORS_DIRECTORY

    def append(self, result: ResultEnvelope) -> Path:
        """Append one result, rejecting a payload that belongs in the detail tier."""

        try:
            result.verify()
        except ResultRecordError as error:
            raise ResultsStoreError(str(error)) from error
        self._reject_committed_detail(result.detail)

        path = self.records_directory / _record_file_name(result)
        payload = canonical_readable_json(result.as_record())
        with self._write_lock():
            self.records_directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() == payload:
                    logger.info("Result %s is already recorded", result.result_id)
                    return path
                raise ResultsStoreError(
                    f"a different result is already recorded at {path}"
                )
            _write_atomically(path, payload)
        logger.info("Recorded result %s in %s", result.result_id, path)
        return path

    def append_bridge(self, bridge: Bridge) -> Path:
        """Record a bridge beside the results it applies to."""

        path = self.bridges_directory / f"{bridge.bridge_id}.json"
        payload = canonical_readable_json(bridge.as_record())
        with self._write_lock():
            self.bridges_directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() == payload:
                    return path
                raise ResultsStoreError(
                    f"a different bridge is already recorded at {path}"
                )
            _write_atomically(path, payload)
        logger.info("Recorded bridge %s in %s", bridge.bridge_id, path)
        return path

    def append_characterization(self, characterization: NoiseCharacterization) -> Path:
        """Record one noise characterization beside the results it qualifies."""

        try:
            characterization.verify()
        except NoiseCharacterizationError as error:
            raise ResultsStoreError(str(error)) from error

        path = self.floors_directory / _characterization_file_name(characterization)
        payload = canonical_readable_json(characterization.as_record())
        with self._write_lock():
            self.floors_directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() == payload:
                    return path
                raise ResultsStoreError(
                    f"a different characterization is already recorded at {path}"
                )
            _write_atomically(path, payload)
        logger.info(
            "Recorded %s noise characterization %s in %s",
            characterization.kind,
            characterization.characterization_id,
            path,
        )
        return path

    def revoke_bridge(self, bridge_id: str) -> Path:
        """Remove a bridge, leaving its removal as a reviewable diff."""

        path = self.bridges_directory / f"{bridge_id}.json"
        if not path.is_file():
            raise ResultsStoreError(f"no bridge is recorded as {bridge_id}")
        with self._write_lock():
            path.unlink()
        logger.info("Revoked bridge %s", bridge_id)
        return path

    def results(self) -> tuple[ResultEnvelope, ...]:
        """Return every recorded result in recording order."""

        envelopes = [
            _load(path, ResultEnvelope)
            for path in sorted(self.records_directory.glob("*.json"))
        ]
        for envelope in envelopes:
            try:
                envelope.verify(recording=False)
            except ResultRecordError as error:
                raise ResultsStoreError(str(error)) from error
        return tuple(
            sorted(
                envelopes,
                key=lambda envelope: (envelope.recorded_at, envelope.result_id),
            )
        )

    def bridges(self) -> tuple[Bridge, ...]:
        """Return every recorded bridge in recording order."""

        bridges = [
            _load(path, Bridge)
            for path in sorted(self.bridges_directory.glob("*.json"))
        ]
        return tuple(
            sorted(bridges, key=lambda bridge: (bridge.recorded_at, bridge.bridge_id))
        )

    def characterizations(self) -> tuple[NoiseCharacterization, ...]:
        """Return every recorded noise characterization in recording order."""

        records = [
            _load(path, NoiseCharacterization)
            for path in sorted(self.floors_directory.glob("*.json"))
        ]
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.recorded_at,
                    record.characterization_id,
                ),
            )
        )

    def _reject_committed_detail(self, detail: DetailReference | None) -> None:
        if detail is None:
            return
        candidate = Path(detail.path)
        if not candidate.is_absolute():
            return
        root = self._root.resolve()
        if root == candidate.resolve() or root in candidate.resolve().parents:
            raise ResultsStoreError(
                "bulk diagnostics must stay in the machine-local detail tier; "
                f"{detail.path} is inside the committed store"
            )

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Hold an exclusive store lock, failing clearly when one is held.

        Two sessions appending at once is the expected conflict, not a rare
        one: separate worktrees run separate benchmarks against one checkout
        of the store. Failing loudly is what keeps a partial write from
        looking like a recorded result.
        """

        self._root.mkdir(parents=True, exist_ok=True)
        lock_path = self._root / LOCK_FILE_NAME
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ResultsStoreError(
                f"another process is writing to {self._root}. If no benchmark "
                f"is running, remove the stale lock at {lock_path}."
            ) from error
        except OSError as error:
            raise ResultsStoreError(
                f"cannot lock the results store at {self._root}: {error}"
            ) from error
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
            os.close(descriptor)
            yield
        finally:
            lock_path.unlink(missing_ok=True)


class DetailStore:
    """Write machine-local bulk diagnostics and return references to them."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Return the detail-tier root."""

        return self._root

    def write(
        self,
        relative_path: str | Path,
        payload: Any,
        *,
        description: str | None = None,
    ) -> DetailReference:
        """Write one diagnostic payload and return its summary-tier reference."""

        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ResultsStoreError(
                f"detail path must stay beneath the detail root: {relative_path}"
            )
        path = self._root / candidate
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = canonical_readable_json(payload)
        _write_atomically(path, encoded)
        return DetailReference(
            path=str(candidate),
            sha256=sha256(encoded).hexdigest(),
            bytes=len(encoded),
            description=description,
        )

    def read(self, reference: DetailReference) -> Any:
        """Read a referenced payload, rejecting bytes that no longer match."""

        path = self._root / reference.path
        if not path.is_file():
            raise ResultsStoreError(f"detail payload does not exist: {path}")
        encoded = path.read_bytes()
        if sha256(encoded).hexdigest() != reference.sha256:
            raise ResultsStoreError(f"detail payload checksum mismatch: {path}")
        return json.loads(encoded)


def resolve_store_root(explicit: str | Path | None = None) -> Path:
    """Resolve the committed store root from an argument or the environment."""

    if explicit is not None:
        return Path(explicit)
    configured = os.environ.get(STORE_ROOT_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(DEFAULT_STORE_DIRECTORY)


def resolve_detail_root(explicit: str | Path | None = None) -> Path:
    """Resolve the machine-local detail root from an argument or the environment."""

    resolved = resolve_optional_detail_root(explicit)
    if resolved is not None:
        return resolved
    raise ResultsStoreError(
        "a detail-tier directory must be provided explicitly, or "
        f"{DETAIL_ROOT_VARIABLE} or ANTHRO_CHESS_RUN_ROOT must be set"
    )


def resolve_optional_detail_root(
    explicit: str | Path | None = None,
) -> Path | None:
    """Resolve a detail root when configured, without making it mandatory."""

    if explicit is not None:
        return Path(explicit)
    configured = os.environ.get(DETAIL_ROOT_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser()
    run_root = os.environ.get("ANTHRO_CHESS_RUN_ROOT", "").strip()
    if run_root:
        return Path(run_root).expanduser() / "benchmark-detail"
    return None


def canonical_readable_json(value: Any) -> bytes:
    """Serialize a record so a diff of the committed tier stays readable."""

    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def checkpoint_labels(results: Iterable[ResultEnvelope]) -> tuple[str, ...]:
    """Return checkpoint labels in the order they were first recorded."""

    ordered: list[str] = []
    for envelope in sorted(
        results,
        key=lambda envelope: (envelope.recorded_at, envelope.result_id),
    ):
        if envelope.checkpoint.label not in ordered:
            ordered.append(envelope.checkpoint.label)
    return tuple(ordered)


def results_for_checkpoint(
    results: Sequence[ResultEnvelope],
    label: str,
) -> tuple[ResultEnvelope, ...]:
    """Return every result recorded for one checkpoint label."""

    return tuple(envelope for envelope in results if envelope.checkpoint.label == label)


def _record_file_name(result: ResultEnvelope) -> str:
    stamp = result.recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{result.kind}-{result.result_id}.json"


def _characterization_file_name(characterization: NoiseCharacterization) -> str:
    stamp = characterization.recorded_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{stamp}-{characterization.kind}-{characterization.characterization_id}.json"
    )


def _write_atomically(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _load(path: Path, schema: type[RecordT]) -> RecordT:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultsStoreError(f"cannot read {path}: {error}") from error
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise ResultsStoreError(f"{path} is not a valid record: {error}") from error
