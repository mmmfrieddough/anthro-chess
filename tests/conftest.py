"""Fixtures shared across the suite, and what it reports about the host.

One module holds them all rather than one per test package, because two
files named ``conftest`` are two modules with the same name to a type
checker, and every fixture here is used from more than one package anyway.
Collection-time facts about the host live in ``accelerators`` instead, since a
``skipif`` condition is evaluated before any fixture exists.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import fields, is_dataclass, replace
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import chess
import pytest
import torch
from tiny_models import tiny_model_config

from anthro_chess.application_logging import (
    APPLICATION_LOGGER_NAME,
    _remove_owned_handlers,
)
from anthro_chess.chess import action_vocabulary_identity, encode_move
from anthro_chess.data import (
    GameEncodingInput,
    SequenceBatch,
    SequenceExample,
    TerminationConfig,
    collate_sequences,
    derive_termination,
    encode_game,
    encoding_identity,
    terminal_action_for,
)
from anthro_chess.data.accounts import account_row_digest
from anthro_chess.data.artifacts import file_sha256
from anthro_chess.data.schema import (
    PREPROCESSING_VERSION,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    derive_game_id,
    encode_clock_remaining_deltas,
    normalized_parquet_schema,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    EnvironmentRecord,
    Measurement,
    ResultEnvelope,
    build_result,
    dataset_reference,
    measurement,
    projection_content_digest,
    registry_snapshot,
    restore_registry,
)
from anthro_chess.evaluation.results.metrics import MOVE_PREDICTION_PROJECTION
from anthro_chess.models import MoveModel
from anthro_chess.models.move_model import model_identity
from anthro_chess.training.checkpoints import save_training_checkpoint

from accelerators import HOST

#: The suite runs sharded, so every worker sharing one machine must be told to
#: stop claiming all of it. Torch sizes its intra-op pool from the core count
#: and knows nothing about the other workers, so N workers each open N threads
#: and the machine ends up oversubscribed by a factor of N. Nothing here is a
#: large enough tensor to win that back: pinned to one thread the suite is
#: faster wall-clock than it is unpinned, and a timing assertion stops
#: measuring how many workers happened to be resident.
torch.set_num_threads(1)


def pytest_report_header() -> list[str]:
    """Report the host's accelerator surface beside what the project accepts.

    Without this, a run exercising no accelerator looks identical on a machine
    that has none and on a fully provisioned one whose accelerator no device
    selection accepts. Both print the same skip count, and only the first makes
    a green run honest about the accelerator.
    """

    return HOST.report_lines()


OPENING_MOVES = (
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5a4",
    "g8f6",
    "e1g1",
    "f8e7",
)


def _action_ids(moves: tuple[str, ...]) -> tuple[int, ...]:
    """Return action ids for UCI move strings."""

    return tuple(encode_move(chess.Move.from_uci(move)) for move in moves)


def _final_board(moves: tuple[str, ...], initial_position: str) -> chess.Board:
    """Return the position a fixture row's moves reach, with its move stack."""

    board = chess.Board(initial_position)
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


def _normalized_row(
    game_id: int,
    *,
    split: str = "test",
    plies: int = 6,
    rating: int | None = 1500,
    ratings: tuple[int | None, int | None] | None = None,
    time_initial_ms: int | None = 300_000,
    time_increment_ms: int | None = 0,
    clocks: bool = True,
    result: str = "1-0",
    moves: tuple[str, ...] | None = None,
    initial_position: str = chess.STARTING_FEN,
    source_date: date | None = date(2019, 6, 15),
) -> dict[str, Any]:
    """Return one canonical normalized game row.

    Explicit ``moves`` and ``initial_position`` let a fixture reach positions
    the shared opening line never visits, such as an available promotion.
    ``ratings`` overrides ``rating`` per player, and the time-control and
    ``source_date`` fields vary independently, so a fixture corpus can span the
    axes a load-time selection or a view filters on.
    """

    moves = OPENING_MOVES[:plies] if moves is None else moves
    white_rating, black_rating = (rating, rating) if ratings is None else ratings
    status = "present"
    # Descending rather than constant so a decoded trace is a plausible clock:
    # a constant one encodes to zeros and would hide a sign error in the delta.
    clock_trace: list[int | None] = (
        [290_000 - 1_000 * index for index in range(len(moves))]
        if clocks
        else [None] * len(moves)
    )
    clock_statuses = [status if clocks else "unavailable"] * len(moves)
    clock_precision_ms = 100 if clocks else None
    # Derive the ending and its terminal action the same way preparation would,
    # so a fixture row never carries a category its own result and moves could
    # not produce, or an action sequence preparation would not have written.
    final_board = _final_board(moves, initial_position)
    termination = derive_termination(
        result=result,
        source_termination="normal",
        final_board=final_board,
        clock_remaining_ms=clock_trace,
        time_initial_ms=time_initial_ms,
        abandonment_clock_share=TerminationConfig().abandonment_clock_share,
    )
    action_ids = list(_action_ids(moves))
    terminal_action_id, terminal_action_status = terminal_action_for(
        termination,
        final_board,
    )
    if terminal_action_id is not None:
        action_ids.append(terminal_action_id)
        clock_trace = [*clock_trace, None]
        clock_statuses.append("unavailable")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": "fixture",
        "source_game_key": f"game{game_id}",
        "source_date": source_date,
        "source_date_status": status if source_date is not None else "unavailable",
        "white_player_digest": account_row_digest(f"white{game_id}"),
        "black_player_digest": account_row_digest(f"black{game_id}"),
        "ruleset": "standard",
        "initial_position": initial_position,
        "result": result,
        "termination": "normal",
        "termination_status": status,
        "termination_category": termination.category.value,
        "termination_by_side_to_move": termination.by_side_to_move,
        "terminal_action_status": terminal_action_status.value,
        "ply_count": len(moves),
        "action_ids": action_ids,
        "white_source_rating": white_rating,
        "white_source_rating_status": status if white_rating else "unavailable",
        "black_source_rating": black_rating,
        "black_source_rating_status": status if black_rating else "unavailable",
        "source_rating_namespace": "fixture_blitz",
        "source_rating_system": "glicko2",
        "white_normalized_rating": white_rating,
        "white_normalized_rating_status": status if white_rating else "unavailable",
        "black_normalized_rating": black_rating,
        "black_normalized_rating_status": status if black_rating else "unavailable",
        "time_initial_ms": time_initial_ms,
        "time_initial_status": status if time_initial_ms is not None else "unavailable",
        "time_increment_ms": time_increment_ms,
        "time_increment_status": (
            status if time_increment_ms is not None else "unavailable"
        ),
        "clock_remaining_delta_ms": encode_clock_remaining_deltas(clock_trace),
        "clock_status": clock_statuses,
        "clock_precision_ms": clock_precision_ms,
        "split": split,
    }


def _write_corpus(
    directory: Path,
    rows: list[dict[str, Any]],
    *,
    source_id: str = "fixture",
    games_per_shard: int | None = None,
    row_group_size: int | None = None,
) -> tuple[Path, Path]:
    """Write a normalized corpus plus a matching manifest, returning both paths.

    ``source_id`` distinguishes separately prepared corpora, whose manifests
    would otherwise be byte-identical here in a way real preparation runs are
    not. ``games_per_shard`` and ``row_group_size`` reproduce the shard and
    row-group layout a real preparation run chooses, which is what the
    shard-backed loader reads and orders against.
    """

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    normalized_directory = directory / "normalized"
    manifest_directory = directory / "manifests"
    normalized_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory.mkdir(parents=True, exist_ok=True)

    per_shard = len(rows) if games_per_shard is None else games_per_shard
    groups = [
        rows[start : start + per_shard] for start in range(0, len(rows), per_shard)
    ]
    shards: list[tuple[Path, list[dict[str, Any]]]] = []
    for index, shard_rows in enumerate(groups):
        name = (
            "games.parquet" if games_per_shard is None else f"games-{index:05d}.parquet"
        )
        games_path = normalized_directory / name
        pq.write_table(
            pa.Table.from_pylist(shard_rows, schema=normalized_parquet_schema()),
            games_path,
            compression="zstd",
            row_group_size=row_group_size,
        )
        shards.append((games_path, shard_rows))

    manifest_path = manifest_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "action_vocabulary": action_vocabulary_identity(),
                "source": {"id": source_id, "version": "v1"},
                "input": {"file_name": "fixture.pgn", "sha256": "0" * 64},
                "split": {"algorithm": "sha256-threshold-v2", "seed": "fixture"},
                "selection": {"algorithm": "fixture"},
                # Counted from moves alone, the way preparation counts it. A
                # training run reads the longest game from here to decide
                # whether the model can encode this corpus at all.
                "games": {
                    "plies": {
                        "maximum_per_game": max(row["ply_count"] for row in rows),
                    },
                },
                "output": {
                    "format": "parquet",
                    "compression": "zstd",
                    "shards": [
                        {
                            "path": f"normalized/{games_path.name}",
                            "sha256": file_sha256(games_path),
                            "games": len(shard_rows),
                            # One entry per split whether or not this shard
                            # holds any, because that is what preparation
                            # writes and a manifest is read against it.
                            "split_counts": {
                                split_name: sum(
                                    row["split"] == split_name for row in shard_rows
                                )
                                for split_name in SPLIT_NAMES
                            },
                        }
                        for games_path, shard_rows in shards
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return normalized_directory, manifest_path


def _write_pinned_archive_config(
    directory: Path,
    *,
    artifact_name: str = "fixture-corpus",
    archive_artifact_name: str = "fixture-archive",
    file_name: str = "fixture.pgn.zst",
) -> Path:
    """Write a selection pinning exactly one archive, returning its path.

    The corpus and the archive are named differently so a caller asserting a
    resolved path shows which of the two it followed.
    """

    config_path = directory / "pinned-archive.toml"
    config_path.write_text(
        f"""
artifact_name = "{artifact_name}"

[source]
id = "fixture"
version = "fixture"
url = "https://example.test/"
license = "CC0-1.0"
rating_namespace_prefix = "lichess"
rating_system = "glicko2"
ratings_are_normalized = true

[[archives]]
artifact_name = "{archive_artifact_name}"
url = "https://example.test/{file_name}"
file_name = "{file_name}"
sha256 = "{"5" * 64}"
compression = "zstd"

[split]
seed = "fixture-split-v1"
validation_fraction = 0.05
test_fraction = 0.05
require_nonempty = true

[filters]
minimum_plies = 1
require_rated = true

[output]
games_per_shard = 50000
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def action_ids() -> Callable[[tuple[str, ...]], tuple[int, ...]]:
    """Return a helper converting UCI move strings into action ids."""

    return _action_ids


def _fixture_game_id(index: int) -> int:
    return derive_game_id("fixture", f"game{index}")


@pytest.fixture
def fixture_game_id() -> Callable[[int], int]:
    """Return the id a fixture row of that index derives to.

    The row no longer stores one, so a test naming a game derives it the way a
    reader does rather than assuming the index is the id.
    """

    return _fixture_game_id


@pytest.fixture
def normalized_row() -> Callable[..., dict[str, Any]]:
    """Return a factory for canonical normalized game rows."""

    return _normalized_row


@pytest.fixture
def write_corpus() -> Callable[..., tuple[Path, Path]]:
    """Return a factory writing a normalized shard plus matching manifest."""

    return _write_corpus


@pytest.fixture
def write_puzzle_artifact() -> Callable[..., Path]:
    """Return a factory writing a loadable puzzle artifact and its manifest."""

    return _write_puzzle_artifact


@pytest.fixture
def write_pinned_archive_config() -> Callable[..., Path]:
    """Return a factory writing a selection that pins exactly one archive."""

    return _write_pinned_archive_config


#: Two verified puzzle lines — an opponent setup move, then one and two
#: solution moves — repeated to fill a generated set. Repeating them keeps
#: first-move accuracy and line completion distinguishable without checking in
#: more chess than a fixture needs.
_PUZZLE_LINES = (
    (
        "N1bk2nr/1p1p1ppp/p2Qp3/8/4P3/6P1/1Pn1KP1P/2qN1B1R b - - 1 14",
        "c2a1 d6f8",
    ),
    (
        "r1bqr1k1/1p2bppp/p4n2/3p2B1/8/2PB1N1P/PP2Q1P1/RN2R1K1 w - - 4 15",
        "b1d2 e7c5 g1h1 e8e2",
    ),
)


def _write_puzzle_artifact(
    directory: Path,
    *,
    ratings: Sequence[int],
    puzzles_per_rating: int,
) -> Path:
    """Write a puzzle set uniform over every exact rating, as a build does.

    The canonical artifact holds the same count at every exact rating in its
    range, and that is the design a subsample has to preserve, so a fixture
    filling ratings unevenly could not tell a correct dial from a biased one.
    """

    rows = sorted(
        f"p{rating:05d}{index:02d},{_PUZZLE_LINES[index % 2][0]},"
        f"{_PUZZLE_LINES[index % 2][1]},{rating},g{rating:05d}{index:02d}"
        for rating in ratings
        for index in range(puzzles_per_rating)
    )
    content = "puzzle_id,initial_fen,moves,rating,source_game_key\n"
    content += "\n".join(rows) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "puzzles.csv").write_text(content, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "name": "fixture-puzzles",
                "version": 1,
                "entries": len(rows),
                "puzzles_sha256": sha256(content.encode()).hexdigest(),
                "source": {"url": "https://example.test/puzzles"},
                "license": {"spdx_id": "CC0-1.0"},
                "selection": {
                    "minimum_rating": 800,
                    "maximum_rating_exclusive": 2800,
                    "local_precision_span": 400,
                },
                "sizing": {},
                "coverage": {},
            }
        ),
        encoding="utf-8",
    )
    return directory


def _sequence_batch(
    *games: tuple[tuple[str, ...], int | None, int | None],
) -> SequenceBatch:
    """Collate encoded games into one padded batch, without inventing targets."""

    examples = []
    for game_offset, (moves, white_rating, black_rating) in enumerate(games):
        board = chess.Board()
        ids = []
        for move_text in moves:
            move = chess.Move.from_uci(move_text)
            assert move in board.legal_moves
            ids.append(encode_move(move))
            board.push(move)
        plies = encode_game(
            GameEncodingInput(
                game_id=100 + game_offset,
                ruleset="standard",
                initial_position=chess.STARTING_FEN,
                action_ids=tuple(ids),
                white_normalized_rating=white_rating,
                black_normalized_rating=black_rating,
                time_initial_ms=None,
                time_increment_ms=None,
                clock_remaining_ms=tuple(None for _ in ids),
            )
        )
        examples.append(
            SequenceExample(
                shard_index=0,
                game_id=100 + game_offset,
                start_ply=0,
                plies=plies,
            )
        )
    return collate_sequences(examples)


@pytest.fixture
def sequence_batch() -> Callable[..., SequenceBatch]:
    """Return a factory building one collated batch from explicit games."""

    return _sequence_batch


#: The training identity a fixture result records for the configuration that
#: produced its weights. Shared by default so two fixture readings compare as
#: two checkpoints of one configuration, which is what a training floor
#: qualifies; a test about the scope passes its own.
FIXTURE_TRAINING_SHA256 = "3c" * 32

FIXTURE_ENVIRONMENT = EnvironmentRecord(
    package_version="0.0.0-fixture",
    git_revision="0" * 40,
    python_version="3.11.0",
    platform="fixture-platform",
    dependencies={"torch": "2.7.0"},
)


@pytest.fixture(autouse=True)
def restored_registry() -> Iterator[None]:
    """Undo any metric registered by a test, keeping the registry global."""

    snapshot = registry_snapshot()
    try:
        yield
    finally:
        restore_registry(snapshot)


@pytest.fixture(autouse=True)
def restored_application_logging() -> Iterator[None]:
    """Undo the level, propagation, and handlers a test left on the process.

    Configuring the application logger is what an entry point wants and what a
    worker with a thousand tests still to run does not, and only sharding
    decides which tests share a worker. Without this, a test reading ``caplog``
    passes or fails on whether an earlier one in its worker happened to raise
    the logger above the level being asserted, since a suppressed record is
    never emitted for any handler to capture. Handlers go through
    ``_remove_owned_handlers`` rather than a restored list so that the ones a
    test opened are closed, and so that a handler pytest attached for the
    current phase is never a candidate.
    """

    application_logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    level = application_logger.level
    propagate = application_logger.propagate
    try:
        yield
    finally:
        _remove_owned_handlers(application_logger)
        application_logger.setLevel(level)
        application_logger.propagate = propagate


def _scored_row(game_id: int, **overrides: Any) -> dict[str, Any]:
    """Return one projected row as a benchmark would hand it to a digest."""

    row: dict[str, Any] = {
        "source_id": "fixture",
        "source_game_key": f"game{game_id}",
        "ruleset": "standard",
        "initial_position": "startpos",
        "action_ids": [1, 2, 3],
        "white_normalized_rating": 1500,
        "black_normalized_rating": 1500,
        # Deliberately outside the move-prediction projection.
        "clock_remaining_delta_ms": [300_000, 299_000],
        "clock_status": ["present", "present"],
        "time_initial_ms": 300_000,
        "time_increment_ms": 0,
        "result": "1-0",
    }
    row.update(overrides)
    return row


@pytest.fixture
def scored_row() -> Callable[..., dict[str, Any]]:
    """Return a factory for one projected row of a scored game."""

    return _scored_row


@pytest.fixture
def move_prediction_component() -> Callable[..., DataComponent]:
    """Return a factory for the move-prediction content digest."""

    def build(rows: Sequence[dict[str, Any]] | None = None) -> DataComponent:
        return projection_content_digest(
            rows if rows is not None else [_scored_row(1), _scored_row(2)],
            MOVE_PREDICTION_PROJECTION,
        )

    return build


@pytest.fixture
def training_scope() -> str:
    """Return the training identity :func:`recorded_result` records by default."""

    return FIXTURE_TRAINING_SHA256


@pytest.fixture
def recorded_result(
    move_prediction_component: Callable[..., DataComponent],
) -> Callable[..., ResultEnvelope]:
    """Return a factory for a complete held-out prediction result."""

    def build(
        *,
        label: str = "checkpoint-a",
        step: int = 8000,
        move_loss: float = 3.5,
        mask_penalty: float = 0.75,
        component: DataComponent | None = None,
        measurements: Sequence[Measurement] | None = None,
        recorded_at: datetime | None = None,
        kind: str = "held-out-prediction",
        training_sha256: str | None = FIXTURE_TRAINING_SHA256,
    ) -> ResultEnvelope:
        data = component if component is not None else move_prediction_component()
        values = (
            list(measurements)
            if measurements is not None
            else [
                measurement("held_out.move_loss", move_loss, data=data),
                measurement("legality.mask_penalty", mask_penalty, data=data),
            ]
        )
        return build_result(
            kind=kind,
            benchmark=BenchmarkReference(name="move-validation", version=1),
            checkpoint=CheckpointReference(
                label=label,
                step=step,
                training_sha256=training_sha256,
            ),
            data=dataset_reference(
                pool_id="fixture-pool",
                pool_version=1,
                view="canonical",
                selected_games=data.games,
                game_ids_sha256="a" * 64,
                components=[data],
            ),
            measurements=values,
            environment=FIXTURE_ENVIRONMENT,
            recorded_at=recorded_at or datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        )

    return build


@pytest.fixture
def training_run() -> Callable[..., Path]:
    """Return a factory writing a retained run and its checkpoint."""

    return write_training_run


@pytest.fixture
def inference_run() -> Callable[..., Path]:
    """Return a factory writing a run with no training corpus recorded.

    What the efficiency benchmarks need is a loadable checkpoint rather than
    a provenance trail: they read no evaluation pool, so nothing about the
    corpus reaches the measurement.
    """

    return write_inference_run


@pytest.fixture
def loadable_run_record() -> dict[str, Any]:
    """Return the run-record fields a load is gated on, all of them current.

    Enough to be reported loadable and no more: a test asking which runs the
    machine report says still load needs the gated fields rather than weights,
    and writing the whole record a training run produces would bury them.
    """

    return {
        "model": model_identity(tiny_model_config()),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": {"precision": "float32", "parameter_dtype": "float32"},
    }


def write_inference_run(path: Path, *, seed: int = 5) -> Path:
    """Write a retained run holding one tiny loadable checkpoint."""

    torch.manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    config = tiny_model_config()
    model = MoveModel(config)
    model_identity = model.identity()
    resolved_config = {
        "config": {"model": config.model_dump(mode="json")},
        "provenance": {"source": None, "overrides": []},
    }
    execution = {
        "device": "cpu",
        "backend": "cpu",
        "precision": "float32",
        "parameter_dtype": "float32",
        "determinism": "strict",
        "gradient_accumulation_steps": 1,
        "phase_profiling": False,
    }
    metadata = {
        "resolved_config": copy.deepcopy(resolved_config),
        "code": {"package_version": "test", "git_revision": "test"},
        "data": {},
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": copy.deepcopy(execution),
    }
    checkpoint = path / "checkpoints" / "step-00000001.pt"
    save_training_checkpoint(
        checkpoint,
        global_step=1,
        counters={"processed_positions": 1},
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state=None,
        scaler_state=None,
        loader_state={},
        compatibility={
            "training_config": {},
            "data": {},
            "model": copy.deepcopy(model_identity),
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
        },
        metadata=metadata,
        device="cpu",
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "version": 3,
                "resolved_config": copy.deepcopy(resolved_config),
                "model": copy.deepcopy(model_identity),
                "action_vocabulary": action_vocabulary_identity(),
                "encoding": encoding_identity(),
                "execution": copy.deepcopy(execution),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint


def write_training_run(
    path: Path,
    *,
    normalized: Path,
    manifest: Path,
    seed: int = 23,
    split: str = "train",
) -> Path:
    """Write a retained run whose provenance names its training corpus.

    ``split`` is which split the run read, which is what the leakage check
    compares against the split a pool was cut from.
    """

    torch.manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    config = tiny_model_config()
    model = MoveModel(config)
    model_identity = model.identity()
    shard = normalized / "games.parquet"
    manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    data_record = {
        "train": {
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            "manifest": manifest_record,
            "normalized_paths": [str(shard.resolve())],
            "dataset_sha256": "0" * 64,
            "loader_configuration_sha256": "1" * 64,
        },
        "validation": None,
    }
    resolved_config = {
        "config": {
            "model": config.model_dump(mode="json"),
            "train": {
                "normalized": str(normalized),
                "manifest": str(manifest),
                "loader": {"split": split, "batch_size": 2},
            },
        },
        "provenance": {"source": None, "overrides": []},
    }
    execution = {
        "device": "cpu",
        "backend": "cpu",
        "precision": "float32",
        "parameter_dtype": "float32",
        "determinism": "strict",
        "gradient_accumulation_steps": 1,
        "phase_profiling": False,
    }
    metadata = {
        "resolved_config": copy.deepcopy(resolved_config),
        "code": {"package_version": "test", "git_revision": "test"},
        "data": copy.deepcopy(data_record),
        "model": copy.deepcopy(model_identity),
        "action_vocabulary": action_vocabulary_identity(),
        "encoding": encoding_identity(),
        "execution": copy.deepcopy(execution),
    }
    checkpoint = path / "checkpoints" / "step-00000001.pt"
    save_training_checkpoint(
        checkpoint,
        global_step=1,
        counters={"processed_positions": 64},
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state=None,
        scaler_state=None,
        loader_state={},
        compatibility={
            "training_config": {},
            "data": {},
            "model": copy.deepcopy(model_identity),
            "action_vocabulary": action_vocabulary_identity(),
            "encoding": encoding_identity(),
        },
        metadata=metadata,
        device="cpu",
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "version": 3,
                "resolved_config": copy.deepcopy(resolved_config),
                "model": copy.deepcopy(model_identity),
                "action_vocabulary": action_vocabulary_identity(),
                "encoding": encoding_identity(),
                "execution": copy.deepcopy(execution),
                # Deliberately different from the checkpoint's own count above:
                # this is the count the run finished on, and holding the two
                # apart is what lets a test see which one a reading reported.
                "optimization": {"processed_positions": 4096},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint


class BlockingReadTensor(torch.Tensor):
    """Stands in for a tensor whose value only a blocking read can answer.

    Asking a real device tensor for a Python value drains its command queue
    before anything else proceeds, and on a CPU-only test run that stall is
    invisible: every check passes and every timing looks the same. This makes
    it loud instead. A boolean context or a scalar read fails outright, while
    moving the tensor to the host first yields an ordinary tensor that answers
    freely, which is the transfer the hot paths are supposed to be reading
    their answers out of.
    """

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: Any = (),
        kwargs: Any = None,
    ) -> Any:
        resolved = kwargs or {}
        result = super().__torch_function__(func, types, args, resolved)
        if not isinstance(result, torch.Tensor):
            return result
        if _transfers_to_host(func, args, resolved):
            return torch.Tensor.as_subclass(result, torch.Tensor)
        return result

    def __bool__(self) -> bool:
        raise AssertionError("a device tensor was evaluated in boolean context")

    def item(self) -> Any:
        raise AssertionError("a device tensor was read one scalar at a time")


def _transfers_to_host(func: Any, args: Any, kwargs: Any) -> bool:
    """Return whether this call is the explicit host read that answers freely."""

    if func is torch.Tensor.cpu:
        return True
    if func is not torch.Tensor.to:
        return False
    return any(
        getattr(value, "type", str(value)) == "cpu"
        for value in (*args[1:], *kwargs.values())
    )


def _device_read_trap(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.Tensor.as_subclass(value, BlockingReadTensor)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _device_read_trap(getattr(value, field.name))
                for field in fields(value)
            },
        )
    return value


@pytest.fixture
def device_read_trap() -> Callable[[Any], Any]:
    """Return a factory rebuilding a batch out of blocking-read tensors."""

    return _device_read_trap
