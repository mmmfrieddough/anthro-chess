"""Whether the policy knows when to resign, read on games humans played.

Resigning is a decision no aggregate over moves can see. A checkpoint that
never resigns and one that resigns while winning post the same move
cross-entropy, so this reading exists to make the difference visible before
resignation is enabled by default.

Everything here comes out of one deterministic pass over frozen human games,
which is what makes it cheap enough to take at a training cadence. What the
model does with the actions when it plays is a property of generated games and
belongs to the rollout, which offers both terminal actions and reads the
termination mix and the three guardrails off the games it already plays.

Two readings, from the same pass:

- the **mass separation**, how much more probability the policy puts on
  resigning at the plies where a human resigned than at the plies where one
  moved. Neither half means much alone, since both rise together on a model
  that has merely learned the action exists;
- the **deficit calibration**, the same mass read against material rather than
  pooled over every ply. A model can spend as much resignation mass as humans
  overall while spending it in the wrong positions, which the separation
  averages away and this catches. Both sides are read at the same plies of the
  same games, so the positions are shared rather than each side bringing its
  own.

Nothing here reports a target rate. The human's own action is the reference,
which is what gives these readings a direction at all.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import StrictBool
from torch import Tensor

from anthro_chess.chess import RESIGNATION_ACTION_ID
from anthro_chess.config import ConfigModel, ResolvedConfig
from anthro_chess.data import SequenceDataLoader
from anthro_chess.data.schema import NormalizedColumn
from anthro_chess.evaluation.execution import runner_device
from anthro_chess.evaluation.policy import (
    POLICY_SCORING_VERSION,
    TerminalActionPolicy,
    active_batch,
    score_terminal_actions,
)
from anthro_chess.evaluation.pool import (
    EvaluationPoolError,
    FrozenPool,
    PoolGenerationPin,
    load_pool,
    pool_rows,
)
from anthro_chess.evaluation.recording import (
    ResultRecording,
    pool_dataset_reference,
    resolve_model,
)
from anthro_chess.evaluation.results import (
    BenchmarkReference,
    CheckpointReference,
    DataComponent,
    DatasetReference,
    Measurement,
    ResultEnvelope,
    measurement,
)
from anthro_chess.evaluation.results.fingerprints import (
    FingerprintError,
    projection_content_digest,
)
from anthro_chess.evaluation.results.metrics import (
    TERMINATION_PREDICTION_PROJECTION,
    TERMINATION_RESIGNATION_CALIBRATION_ERROR,
    TERMINATION_RESIGNATION_CALIBRATION_GAP,
    TERMINATION_RESIGNATION_MASS_AT_MOVES,
    TERMINATION_RESIGNATION_MASS_AT_RESIGNATION,
    TERMINATION_RESIGNATION_MASS_SEPARATION,
)
from anthro_chess.evaluation.scoring import (
    SCORED_COLUMNS,
    EvaluationLoaderConfig,
    ScoringError,
    ScoringInputs,
    build_scoring_inputs,
    rows_identity_sha256,
)
from anthro_chess.evaluation.selection import CheckpointSelection
from anthro_chess.evaluation.slices import board_from_encoding, material_balance
from anthro_chess.evaluation.views import ViewConfig, ViewSelection, apply_view
from anthro_chess.models import MoveModelBatch
from anthro_chess.runtime import ActionModelRunner

logger = logging.getLogger(__name__)

TERMINATION_BENCHMARK_VERSION = 2

TERMINATION_KIND = "game-termination"
TERMINATION_BENCHMARK = BenchmarkReference(
    name="game-termination",
    version=TERMINATION_BENCHMARK_VERSION,
)

#: What the pass reads from a pool row: the encoder's own inputs, plus the two
#: ending columns the projection this reading is identified by declares.
_SCORED_COLUMNS = (
    *SCORED_COLUMNS,
    NormalizedColumn.TERMINATION_CATEGORY.value,
    NormalizedColumn.TERMINAL_ACTION_STATUS.value,
)

#: Deficit buckets, in pawns behind from the point of view of the player to
#: move. Chosen against the population decision 0017 measured, where
#: resignations sat at a median of six pawns down with 73% at least three down:
#: the boundaries separate resigning while not behind, resigning a pawn or two
#: down, and the ordinary range, rather than resolving the far tail nobody
#: reads.
DEFICIT_BUCKET_EDGES: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 9.0)

#: The bands those edges cut, in deficit order, which is the order they are
#: reported in. Named once so the label a deficit lands on and the label the
#: reading walks cannot disagree: they did, a populated band would vanish from
#: the calibration rather than fail.
DEFICIT_BUCKETS: tuple[str, ...] = (
    f"below-{DEFICIT_BUCKET_EDGES[0]:g}",
    *(
        f"{low:g}-to-{high:g}"
        for low, high in zip(
            DEFICIT_BUCKET_EDGES, DEFICIT_BUCKET_EDGES[1:], strict=False
        )
    ),
    f"{DEFICIT_BUCKET_EDGES[-1]:g}-and-above",
)


class TerminationBenchmarkError(ValueError):
    """Raised when resignation prediction cannot be measured as configured."""


class ScoringModelRunner(Protocol):
    """The runner surface this reading needs, which is one forward pass."""

    def action_logits(self, batch: MoveModelBatch) -> Tensor:
        """Return raw action logits for one aligned evaluation batch."""


class TerminationDetailConfig(ConfigModel):
    """What the machine-local detail tier keeps beside a committed summary."""

    #: Per-decision resignation mass. One row per scored ply, so it is the
    #: largest payload this benchmark can produce and the one that makes an
    #: unexpected reading explainable.
    retain_decisions: StrictBool = True


class TerminationBenchmarkConfig(CheckpointSelection, PoolGenerationPin):
    """Code-owned schema for ``anthro eval termination``."""

    #: A frozen evaluation pool. The reading is entirely a pass over human
    #: games, so a suite without one has nothing to score.
    pool: Path
    #: Every ply of every selected game is scored, so this view is sized by
    #: scoring cost. The plies where a human resigned are the scarce half:
    #: roughly three games in ten carry one, and they are what the calibration
    #: buckets are estimated from.
    view: ViewConfig = ViewConfig(name="termination-held-out", maximum_games=4096)
    loader: EvaluationLoaderConfig = EvaluationLoaderConfig()
    detail: TerminationDetailConfig = TerminationDetailConfig()


@dataclass(frozen=True)
class CalibrationBucket:
    """One material band, and what each side did with the positions in it."""

    bucket: str
    #: Scored plies whose position sat in this band, which is the denominator
    #: both sides share.
    plies: int
    human_resignations: int
    #: Mean probability the policy put on resigning across the same plies.
    model_mass: float

    @property
    def human_rate(self) -> float:
        """Return the share of those plies where the human resigned."""

        return self.human_resignations / self.plies

    @property
    def gap(self) -> float:
        """Return how much more mass the policy spent than humans did."""

        return self.model_mass - self.human_rate

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of one calibration band."""

        return {
            "bucket": self.bucket,
            "plies": self.plies,
            "human_resignations": self.human_resignations,
            "human_rate": self.human_rate,
            "model_mass": self.model_mass,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class ResignationCalibration:
    """The policy's resignation mass against the human rate, by deficit."""

    buckets: tuple[CalibrationBucket, ...]

    @property
    def plies(self) -> int:
        """Return how many scored plies the calibration was built from."""

        return sum(bucket.plies for bucket in self.buckets)

    @property
    def human_resignations(self) -> int:
        """Return how many of those plies a human resigned at."""

        return sum(bucket.human_resignations for bucket in self.buckets)

    @property
    def error(self) -> float | None:
        """Return the ply-weighted absolute gap between the two sides.

        Weighted by how often a position in each band comes up rather than
        averaged over bands, so this reads as the gap at a ply drawn at random.
        An unweighted mean would let a band holding thirty plies count for as
        much as one holding twenty thousand.
        """

        return self._weighted(lambda bucket: abs(bucket.gap))

    @property
    def gap(self) -> float | None:
        """Return the same comparison signed rather than absolute."""

        return self._weighted(lambda bucket: bucket.gap)

    def _weighted(self, value: Callable[[CalibrationBucket], float]) -> float | None:
        """Return one ply-weighted reduction, or nothing to weight it against.

        A view holding no human resignation gives every band a rate of zero,
        against which any mass the policy spends reads as spending more than
        humans did. That is a missing reference rather than a measured excess,
        and it is the condition that withholds the mass at resignation too.
        """

        plies = self.plies
        if not plies or not self.human_resignations:
            return None
        return sum(value(bucket) * bucket.plies for bucket in self.buckets) / plies

    def as_record(self) -> dict[str, Any]:
        """Return the stored form of the whole calibration."""

        return {
            "bucket_edges": list(DEFICIT_BUCKET_EDGES),
            "plies": self.plies,
            "error": self.error,
            "gap": self.gap,
            "buckets": [bucket.as_record() for bucket in self.buckets],
        }


@dataclass(frozen=True)
class HeldOutResignation:
    """What the policy says about resigning at positions humans reached."""

    view: ViewSelection
    dataset: DatasetReference
    #: The projected content this pass scored. Series identity rather than
    #: provenance: nothing was generated here, so what the reading is scoped by
    #: is exactly the human decisions it read.
    data: DataComponent
    games: int
    resignation_plies: int
    move_plies: int
    mass_at_resignation: float | None
    mass_at_moves: float | None
    calibration: ResignationCalibration
    unavailable: dict[str, str] = field(default_factory=dict)
    decisions: tuple[TerminalActionPolicy, ...] = field(default=(), repr=False)

    @property
    def separation(self) -> float | None:
        """Return how much more mass sits where humans resigned."""

        if self.mass_at_resignation is None or self.mass_at_moves is None:
            return None
        return self.mass_at_resignation - self.mass_at_moves

    def as_record(self) -> dict[str, Any]:
        """Return the payload stored in the detail tier."""

        return {
            "policy_scoring_version": POLICY_SCORING_VERSION,
            "view": self.view.as_record(),
            "games": self.games,
            "resignation_plies": self.resignation_plies,
            "move_plies": self.move_plies,
            "mass_at_resignation": self.mass_at_resignation,
            "mass_at_moves": self.mass_at_moves,
            "separation": self.separation,
            "calibration": self.calibration.as_record(),
            "unavailable": dict(sorted(self.unavailable.items())),
            "decisions": [decision.as_record() for decision in self.decisions],
        }


@dataclass(frozen=True)
class TerminationBenchmarkResult:
    """Everything one resignation reading measured, and where it was written."""

    checkpoint: CheckpointReference
    held_out: HeldOutResignation
    envelopes: tuple[ResultEnvelope, ...] = ()
    recorded_paths: tuple[Path, ...] = ()
    detail_paths: tuple[Path, ...] = ()

    def as_record(self) -> dict[str, Any]:
        """Return the full structured result, detail tier included."""

        return {
            "version": TERMINATION_BENCHMARK_VERSION,
            "checkpoint": self.checkpoint.model_dump(mode="json"),
            "held_out": self.held_out.as_record(),
            "recorded": [str(path) for path in self.recorded_paths],
        }


def benchmark_termination(
    resolved_config: ResolvedConfig[TerminationBenchmarkConfig],
    *,
    run_root: Path | None = None,
    recording: ResultRecording,
    runner: ActionModelRunner | None = None,
    checkpoint: CheckpointReference | None = None,
) -> TerminationBenchmarkResult:
    """Score frozen human games for what the policy says about resigning.

    Passing no ``store`` measures everything and records nothing, which is what
    an exploratory reading wants.

    A ``runner`` may be supplied to measure an already-loaded checkpoint, in
    which case ``checkpoint`` identifies it. Otherwise both are resolved from
    the configuration.
    """

    config = resolved_config.value
    loaded, identity = resolve_model(
        config.model,
        runner,
        checkpoint,
        label=config.checkpoint_label,
        run_root=run_root,
        error=TerminationBenchmarkError,
    )
    pool = _load_pool(config)
    result = TerminationBenchmarkResult(
        checkpoint=identity,
        held_out=_held_out_resignation(config, loaded, pool),
    )
    _record(result, recording)
    return result


def _held_out_resignation(
    config: TerminationBenchmarkConfig,
    runner: ActionModelRunner,
    pool: FrozenPool,
) -> HeldOutResignation:
    """Score every ply of the selected view for its resignation mass."""

    selection = apply_view(pool.games, config.view)
    if not selection.game_ids:
        raise TerminationBenchmarkError(
            f"view {config.view.name!r} selected no human games to score"
        )
    rows = _rows_for(pool, selection.game_ids)
    scorer = _scoring_runner(runner)
    inputs, decisions = _score_decisions(config, scorer, rows, selection)
    component = _projection_component(rows)

    at_resignation = [
        decision.resignation_mass
        for decision in decisions
        if decision.target_action_id == RESIGNATION_ACTION_ID
    ]
    at_moves = [
        decision.resignation_mass
        for decision in decisions
        if not decision.target_is_terminal
    ]
    unavailable: dict[str, str] = {}
    if not at_resignation:
        unavailable["resignation_mass_at_resignation"] = (
            "no game in the view carries a resignation action, so the policy "
            "has no human resignation to be scored against"
        )
    calibration = _calibration(decisions, inputs)
    if calibration.error is None:
        unavailable["resignation_calibration"] = (
            "no scored decision could be matched to the position it was taken "
            "at, so there is no material to read the mass against"
            if not calibration.plies
            else "no game in the view carries a resignation action, so every "
            "band's human rate is zero and the policy has nothing to be "
            "calibrated against"
        )
    games = len({decision.game_id for decision in decisions})
    logger.info(
        "Resignation prediction: %s game(s), %s resignation ply/plies of %s",
        games,
        len(at_resignation),
        len(decisions),
    )
    return HeldOutResignation(
        view=selection,
        dataset=_dataset_reference(pool, selection, rows),
        data=component,
        games=games,
        resignation_plies=len(at_resignation),
        move_plies=len(at_moves),
        mass_at_resignation=_mean(at_resignation),
        mass_at_moves=_mean(at_moves),
        calibration=calibration,
        unavailable=unavailable,
        decisions=tuple(decisions) if config.detail.retain_decisions else (),
    )


def _score_decisions(
    config: TerminationBenchmarkConfig,
    runner: ScoringModelRunner,
    rows: Sequence[Mapping[str, Any]],
    selection: ViewSelection,
) -> tuple[ScoringInputs, tuple[TerminalActionPolicy, ...]]:
    """Run one deterministic pass over the view and keep the terminal mass.

    The encoded positions come back with the scores because the calibration
    reads the material at each of them. Replaying the rows again would be a
    second answer to which position sits at one decision, and the encoder is
    the one that decided which plies were scored at all.
    """

    try:
        inputs = build_scoring_inputs(
            rows,
            split="test",
            batch_size=config.loader.batch_size,
            length_bucket_width=config.loader.length_bucket_width,
            identity_sha256=rows_identity_sha256(rows, context=selection.name),
        )
    except ScoringError as error:
        raise TerminationBenchmarkError(str(error)) from error

    device = runner_device(runner)
    scored: list[TerminalActionPolicy] = []
    loader = SequenceDataLoader(inputs.dataset, inputs.loader_config)
    for sequence_batch in loader:
        batch = MoveModelBatch.from_sequence_batch(sequence_batch, device=device)
        scored.extend(
            score_terminal_actions(active_batch(runner.action_logits(batch), batch))
        )
    if not scored:
        raise TerminationBenchmarkError(
            "the configured view selected no positions to score"
        )
    return inputs, tuple(scored)


def _calibration(
    decisions: Sequence[TerminalActionPolicy],
    inputs: ScoringInputs,
) -> ResignationCalibration:
    """Bucket the scored plies by material and compare the two sides.

    The material a player was behind by is the dependency-free position-quality
    signal decision 0017 settled on, and reading both sides at the same plies
    is what makes the comparison a comparison: the model's mass and the human's
    own choice are two answers to one position rather than two populations.
    """

    plies: dict[str, int] = {}
    resignations: dict[str, int] = {}
    mass: dict[str, float] = {}
    for decision in decisions:
        encoding = inputs.plies.get((decision.game_id, decision.ply_index))
        if encoding is None:
            continue
        board = board_from_encoding(encoding.board)
        name = _bucket(float(-material_balance(board, board.turn)))
        plies[name] = plies.get(name, 0) + 1
        mass[name] = mass.get(name, 0.0) + decision.resignation_mass
        if decision.target_action_id == RESIGNATION_ACTION_ID:
            resignations[name] = resignations.get(name, 0) + 1
    return ResignationCalibration(
        buckets=tuple(
            CalibrationBucket(
                bucket=name,
                plies=plies[name],
                human_resignations=resignations.get(name, 0),
                model_mass=mass[name] / plies[name],
            )
            for name in DEFICIT_BUCKETS
            if name in plies
        )
    )


def _load_pool(config: TerminationBenchmarkConfig) -> FrozenPool:
    """Load the frozen pool this reading is measured against."""

    try:
        return load_pool(
            config.pool,
            expected_game_ids_sha256=config.expected_pool_game_ids_sha256,
        )
    except EvaluationPoolError as error:
        raise TerminationBenchmarkError(str(error)) from error


def _rows_for(pool: FrozenPool, game_ids: Sequence[int]) -> tuple[dict[str, Any], ...]:
    """Return the selected pool rows, in ascending game-id order."""

    return pool_rows(
        pool,
        game_ids,
        _SCORED_COLUMNS,
        error=TerminationBenchmarkError,
    )


def _dataset_reference(
    pool: FrozenPool,
    selection: ViewSelection,
    rows: Sequence[Mapping[str, Any]],
) -> DatasetReference:
    """Describe the human games this reading read."""

    return pool_dataset_reference(
        pool,
        selection,
        _projection_component(rows),
        error=TerminationBenchmarkError,
    )


def _projection_component(rows: Sequence[Mapping[str, Any]]) -> DataComponent:
    """Return the digest of the content this reading actually consumed."""

    try:
        return projection_content_digest(rows, TERMINATION_PREDICTION_PROJECTION)
    except FingerprintError as error:
        raise TerminationBenchmarkError(str(error)) from error


def _record(
    result: TerminationBenchmarkResult,
    recording: ResultRecording,
) -> None:
    """Write the one envelope this reading produces."""

    recorder = recording.measuring(
        result.checkpoint,
        kind=TERMINATION_KIND,
        benchmark=TERMINATION_BENCHMARK,
    )
    held_out = result.held_out
    # A pass that measured nothing still writes its payload, which is what
    # `add` does with empty measurements: the reading itself is the evidence
    # for why there is nothing to commit.
    recorder.add(
        _held_out_measurements(held_out),
        payload=held_out.as_record,
        description="Held-out resignation prediction",
        slug="held-out-resignation",
        # Series identity here, not provenance: this is a deterministic pass
        # over fixed content and generated nothing, so the content is what it
        # is scoped by.
        data=held_out.dataset,
    )


def _held_out_measurements(reading: HeldOutResignation) -> tuple[Measurement, ...]:
    """Return the readings the summary tier records.

    Sample sizes are per metric rather than shared. The mass readings are
    averaged over populations that differ by orders of magnitude — a handful of
    resignation plies against every move ply in the view — and reporting the
    larger count for both would overstate the first one's precision to every
    reader, the noise-floor layer included.
    """

    data = reading.data
    calibration = reading.calibration
    values: tuple[tuple[str, float | None, int], ...] = (
        (
            TERMINATION_RESIGNATION_MASS_AT_RESIGNATION.identifier,
            reading.mass_at_resignation,
            reading.resignation_plies,
        ),
        (
            TERMINATION_RESIGNATION_MASS_AT_MOVES.identifier,
            reading.mass_at_moves,
            reading.move_plies,
        ),
        (
            TERMINATION_RESIGNATION_MASS_SEPARATION.identifier,
            reading.separation,
            reading.resignation_plies,
        ),
        (
            TERMINATION_RESIGNATION_CALIBRATION_ERROR.identifier,
            calibration.error,
            calibration.plies,
        ),
        (
            TERMINATION_RESIGNATION_CALIBRATION_GAP.identifier,
            calibration.gap,
            calibration.plies,
        ),
    )
    return tuple(
        measurement(
            identifier,
            value,
            data=data,
            sample_size=sample_size if sample_size else None,
        )
        for identifier, value, sample_size in values
        if value is not None
    )


def _scoring_runner(runner: ActionModelRunner) -> ScoringModelRunner:
    """Return the runner as the narrower surface this pass needs."""

    if not callable(getattr(runner, "action_logits", None)):
        raise TerminationBenchmarkError(
            "this reading needs a runner that scores whole batches, and the "
            "loaded model cannot score stored positions"
        )
    return runner  # type: ignore[return-value]


def _bucket(deficit: float) -> str:
    """Return the band one deficit falls in, by where it sits on the edges."""

    return DEFICIT_BUCKETS[bisect_right(DEFICIT_BUCKET_EDGES, deficit)]


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean, or ``None`` when nothing was averaged."""

    return sum(values) / len(values) if values else None


__all__ = [
    "DEFICIT_BUCKETS",
    "DEFICIT_BUCKET_EDGES",
    "TERMINATION_BENCHMARK",
    "TERMINATION_BENCHMARK_VERSION",
    "TERMINATION_KIND",
    "CalibrationBucket",
    "HeldOutResignation",
    "ResignationCalibration",
    "ScoringModelRunner",
    "TerminationBenchmarkConfig",
    "TerminationBenchmarkError",
    "TerminationBenchmarkResult",
    "TerminationDetailConfig",
    "benchmark_termination",
]
