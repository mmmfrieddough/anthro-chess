"""Separating decisions the model got wrong from decisions sampling got wrong.

Two decisions in one played game failed for opposite reasons. In the first the
model ranked the three winning captures first, second, and third, and the draw
took its seventh choice. In the second it ranked a free queen capture sixth and
the draw took its own top choice. Lowering the temperature would have prevented
the first and guaranteed the second, so no aggregate that reports only how the
game turned out can say which fix a checkpoint needs.

This layer reports both classes side by side over the decisions a run actually
made. Every quantity comes from the untempered policy, which is what the
runtime records at selection time, so a reading describes the model rather than
the dial. Whether the draw followed the model is reported separately, and the
temperature it was drawn at is carried on every cell.

Nothing here consults a model. Samples come either from records a run already
wrote or from :func:`score_played_decisions`, which re-scores a move sequence
the runtime did not originate through the same selection path. Choosing a
default temperature is out of scope; this is the measurement such a choice
would need.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.games.records import (
    DecisionPolicy,
    GameRecord,
    GameRecordError,
    PlayerSlot,
    SeatRecord,
    parse_game_records,
)
from anthro_chess.evaluation.results import DataComponent, Measurement, measurement
from anthro_chess.evaluation.results.metrics import (
    DECISION_DEPARTURE_POLICY_REGRET,
    DECISION_POLICY_REGRET,
    DECISION_PREFERRED_PROBABILITY,
    DECISION_PREFERRED_SELECTION_RATE,
    DECISION_SELECTED_RANK,
)
from anthro_chess.inference import ModelRunnerConfig
from anthro_chess.runtime import (
    ActionModelRunner,
    DecisionRuntimeError,
    GameSession,
    RuntimeConfig,
    SelectionPolicy,
)

DECISION_DECOMPOSITION_VERSION = 1

#: Which distribution every probability in a decomposition comes from, declared
#: in the artifact rather than left to a reader. The tempered distribution the
#: draw used and the model's own distribution over enabled actions differ most
#: exactly where this measurement is interesting, so an artifact that did not
#: say which one it holds would be unreadable a year later.
POLICY_DISTRIBUTION = "untempered_enabled_actions"

logger = logging.getLogger(__name__)


class DecisionDecompositionError(ValueError):
    """Raised when a decomposition cannot be computed as requested."""


class DecisionAnalysisConfig(ConfigModel):
    """Code-owned schema for ``anthro eval decisions``.

    Only the checkpoint is configured. What to analyze is an argument rather
    than a setting: a decomposition is read over one payload or one session log
    at a time, and a run does not accumulate a declared workload the way a
    benchmark does.
    """

    model: ModelRunnerConfig = ModelRunnerConfig()


@dataclass(frozen=True)
class DecisionSetting:
    """The dials one decision was made under.

    Rating and temperature are separate axes because the balance between the
    two error classes is expected to depend on both, and because a run that
    varies either produces cells that must not be averaged together.
    """

    target_rating: int | None
    temperature: float | None

    @property
    def label(self) -> str:
        """Return the short human-readable cell name a report prints."""

        rating = "unset" if self.target_rating is None else str(self.target_rating)
        temperature = "unset" if self.temperature is None else f"{self.temperature:g}"
        return f"rating {rating}, temperature {temperature}"

    @property
    def sort_key(self) -> tuple[bool, int, bool, float]:
        """Return a total order, so cell output does not depend on input order."""

        return (
            self.target_rating is None,
            self.target_rating or 0,
            self.temperature is None,
            self.temperature or 0.0,
        )

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable cell coordinates."""

        return {
            "target_rating": self.target_rating,
            "temperature": self.temperature,
        }


@dataclass(frozen=True)
class DecisionSample:
    """One decision, in the terms the decomposition classifies it by.

    The probabilities are the model's own untempered distribution over the
    actions the position enabled. ``temperature`` is recorded beside them
    rather than applied to them.
    """

    game_id: str
    ply_index: int
    slot: PlayerSlot
    action_id: int
    setting: DecisionSetting
    enabled_action_count: int
    selected_probability: float
    selected_rank: int
    preferred_action_id: int
    preferred_probability: float

    @property
    def followed_policy(self) -> bool:
        """Return whether the selection was one the model preferred.

        Decided by rank rather than by action identity. When two actions carry
        the same logit the model has no preference between them, so a draw that
        took the one the argmax did not name overrode nothing.
        """

        return self.selected_rank == 1

    @property
    def policy_regret(self) -> float:
        """Return the untempered probability the selection gave up.

        Probability rather than a log ratio: it is bounded, it is what the
        motivating measurement used, and a log gap on a near-zero-probability
        action reports a large number for a decision the model barely
        considered either way.
        """

        return max(self.preferred_probability - self.selected_probability, 0.0)

    def as_record(self) -> dict[str, Any]:
        """Return the detail-tier record for one decision."""

        return {
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "slot": self.slot,
            "action_id": self.action_id,
            "setting": self.setting.as_record(),
            "enabled_action_count": self.enabled_action_count,
            "selected_probability": self.selected_probability,
            "selected_rank": self.selected_rank,
            "preferred_action_id": self.preferred_action_id,
            "preferred_probability": self.preferred_probability,
            "followed_policy": self.followed_policy,
            "policy_regret": self.policy_regret,
        }


@dataclass(frozen=True)
class DecisionSet:
    """Every decision one analysis pass could classify, and what it could not.

    ``unscored_decisions`` is reported rather than dropped silently. A random
    or external-engine seat has no policy to decompose, and a reader has to be
    able to tell a suite where half the decisions were unscorable from one
    where every decision was measured.
    """

    samples: tuple[DecisionSample, ...]
    unscored_decisions: int = 0


@dataclass(frozen=True)
class DecisionCell:
    """The decomposition over one setting, or over a whole pass.

    Both classes are reported as their own quantity rather than left to be
    inferred from each other, and the regret over departures is reported apart
    from the pooled regret: a small pooled figure can mean either that the draw
    rarely overrode the model or that it did so only on near ties.
    """

    setting: DecisionSetting | None
    decisions: int
    followed_policy: int
    departures: int
    preferred_selection_rate: float
    policy_regret: float
    departure_policy_regret: float
    maximum_policy_regret: float
    preferred_probability: float
    selected_probability: float
    selected_rank: float
    enabled_action_count: float

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable record for one cell."""

        return {
            "setting": None if self.setting is None else self.setting.as_record(),
            "decisions": self.decisions,
            "followed_policy": self.followed_policy,
            "departures": self.departures,
            "preferred_selection_rate": self.preferred_selection_rate,
            "policy_regret": self.policy_regret,
            "departure_policy_regret": self.departure_policy_regret,
            "maximum_policy_regret": self.maximum_policy_regret,
            "preferred_probability": self.preferred_probability,
            "selected_probability": self.selected_probability,
            "selected_rank": self.selected_rank,
            "enabled_action_count": self.enabled_action_count,
        }


@dataclass(frozen=True)
class DecisionDecomposition:
    """One decomposition: the pooled reading, its cells, and its decisions.

    The per-decision samples are retained rather than summarized away. A
    checkpoint's interesting decisions are individual ones — the drawn seventh
    choice, the free queen ranked sixth — and a reader cannot recover them from
    a mean. They are detail-tier bulk by the storage rules, so they travel in
    this record and never into a committed measurement.
    """

    version: int
    overall: DecisionCell
    cells: tuple[DecisionCell, ...]
    unscored_decisions: int
    samples: tuple[DecisionSample, ...] = ()

    @property
    def settings(self) -> tuple[DecisionSetting, ...]:
        """Return the distinct settings this pass measured, in cell order."""

        return tuple(cell.setting for cell in self.cells if cell.setting is not None)

    def cell(self, setting: DecisionSetting) -> DecisionCell:
        """Return one setting's cell."""

        for candidate in self.cells:
            if candidate.setting == setting:
                return candidate
        raise DecisionDecompositionError(
            f"this decomposition has no cell for {setting.label}"
        )

    def reference_cell(self) -> DecisionCell:
        """Return the cell a committed measurement may be taken from.

        A pass that varied rating or temperature has no such cell. Its pooled
        reading is a blend whose value would move with grid composition rather
        than with the model, so a caller that varies the dials has to name the
        cell it means.
        """

        if len(self.cells) != 1:
            raise DecisionDecompositionError(
                "this pass varied rating or temperature over "
                f"{len(self.cells)} setting(s), so it has no single reference "
                "cell; name the setting to report"
            )
        return self.cells[0]

    def as_record(self) -> dict[str, Any]:
        """Return the machine-readable decomposition, per decision included."""

        return {
            "version": self.version,
            "policy_distribution": POLICY_DISTRIBUTION,
            "overall": self.overall.as_record(),
            "cells": [cell.as_record() for cell in self.cells],
            "unscored_decisions": self.unscored_decisions,
            "samples": [sample.as_record() for sample in self.samples],
        }


def collect_decisions(records: Iterable[GameRecord]) -> DecisionSet:
    """Return the classifiable decisions of already-recorded games.

    Reads retained records rather than replaying anything, so a decomposition
    over a rollout suite costs a pass over files the suite already wrote.
    """

    samples: list[DecisionSample] = []
    unscored = 0
    for record in records:
        settings = {
            slot: _seat_setting(record.seat(slot)) for slot in ("white", "black")
        }
        for decision in record.decisions:
            if decision.policy is None:
                unscored += 1
                continue
            samples.append(
                _sample(
                    game_id=record.game_id,
                    ply_index=decision.ply_index,
                    slot=decision.slot,
                    action_id=decision.action_id,
                    setting=settings[decision.slot],
                    policy=decision.policy,
                )
            )
    return DecisionSet(samples=tuple(samples), unscored_decisions=unscored)


def decompose_game_records(path: Path) -> DecisionDecomposition:
    """Decompose a stored payload of generated games.

    Costs no model pass at all: a harness run recorded each decision's policy
    when it made it, so this reads the payload the suite already wrote.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DecisionDecompositionError(
            f"cannot read game records from {path}: {error}"
        ) from error
    try:
        records = parse_game_records(payload)
    except (GameRecordError, ValueError) as error:
        raise DecisionDecompositionError(
            f"{path} is not a readable game payload: {error}"
        ) from error
    return summarize_decisions(collect_decisions(records))


def summarize_decisions(decisions: DecisionSet) -> DecisionDecomposition:
    """Aggregate classified decisions into a pooled reading and its cells."""

    if not decisions.samples:
        raise DecisionDecompositionError(
            "a decomposition needs at least one classifiable decision"
        )
    grouped: dict[DecisionSetting, list[DecisionSample]] = {}
    for sample in decisions.samples:
        grouped.setdefault(sample.setting, []).append(sample)
    return DecisionDecomposition(
        version=DECISION_DECOMPOSITION_VERSION,
        overall=_cell(None, decisions.samples),
        cells=tuple(
            _cell(setting, grouped[setting])
            for setting in sorted(grouped, key=lambda setting: setting.sort_key)
        ),
        unscored_decisions=decisions.unscored_decisions,
        samples=decisions.samples,
    )


def decision_measurements(
    cell: DecisionCell,
    *,
    data: DataComponent,
) -> tuple[Measurement, ...]:
    """Return the committed measurements for one cell of a decomposition.

    One cell rather than a whole decomposition, because a result reports each
    metric once and a pooled figure over a rating or temperature grid would
    move with the grid. A suite passes the cell for its declared reference
    setting.

    The regret over departures is omitted when there were none. A run at
    temperature zero has no departures at all, and reporting zero would read as
    a measured near tie rather than as a quantity that does not exist.
    """

    values = [
        measurement(
            DECISION_PREFERRED_SELECTION_RATE.identifier,
            cell.preferred_selection_rate,
            data=data,
            sample_size=cell.decisions,
        ),
        measurement(
            DECISION_POLICY_REGRET.identifier,
            cell.policy_regret,
            data=data,
            sample_size=cell.decisions,
        ),
        measurement(
            DECISION_PREFERRED_PROBABILITY.identifier,
            cell.preferred_probability,
            data=data,
            sample_size=cell.decisions,
        ),
        measurement(
            DECISION_SELECTED_RANK.identifier,
            cell.selected_rank,
            data=data,
            sample_size=cell.decisions,
        ),
    ]
    if cell.departures:
        values.append(
            measurement(
                DECISION_DEPARTURE_POLICY_REGRET.identifier,
                cell.departure_policy_regret,
                data=data,
                sample_size=cell.departures,
            )
        )
    return tuple(sorted(values, key=lambda value: value.metric))


@dataclass(frozen=True)
class PlayedDecision:
    """One decision to re-score, with the exact history it was made from.

    The whole history is carried rather than only the position, because the
    model conditions on the trajectory: two identical boards reached by
    different move orders are different decisions to it.
    """

    game_id: str
    ply_index: int
    initial_fen: str
    history: tuple[chess.Move, ...]
    action_id: int
    config: RuntimeConfig

    @property
    def setting(self) -> DecisionSetting:
        """Return the dials this decision was played under."""

        return DecisionSetting(
            target_rating=self.config.target_rating,
            temperature=self.config.temperature,
        )


def score_played_decisions(
    runner: ActionModelRunner,
    decisions: Sequence[PlayedDecision],
) -> DecisionSet:
    """Re-score decisions a run made without recording their policy.

    This is what makes a manually played game analyzable with the same code
    path as a benchmark rollout: given the move sequence and the settings it
    was played under, the policy behind each decision is recovered by asking
    the same session the runtime would have used.

    Decisions are scored in the order given, through one session per stretch of
    unchanged settings, so an ordinary game reuses its encoded prefix instead of
    resynchronizing from the root at every ply.
    """

    samples: list[DecisionSample] = []
    session: GameSession | None = None
    active: RuntimeConfig | None = None
    for decision in decisions:
        if session is None or active != decision.config:
            session = GameSession(runner, config=decision.config)
            active = decision.config
        try:
            session.sync_position(
                initial_fen=decision.initial_fen,
                moves=decision.history,
            )
            policy = session.score_action(decision.action_id)
        except DecisionRuntimeError as error:
            raise DecisionDecompositionError(
                f"cannot score {decision.game_id} at ply {decision.ply_index}: {error}"
            ) from error
        samples.append(
            _sample(
                game_id=decision.game_id,
                ply_index=decision.ply_index,
                slot=_slot(session.board.turn),
                action_id=decision.action_id,
                setting=decision.setting,
                policy=policy,
            )
        )
    logger.info("Re-scored %s recorded decision(s)", len(samples))
    return DecisionSet(samples=tuple(samples))


def _sample(
    *,
    game_id: str,
    ply_index: int,
    slot: PlayerSlot,
    action_id: int,
    setting: DecisionSetting,
    policy: DecisionPolicy | SelectionPolicy,
) -> DecisionSample:
    """Return one sample from either producer's policy record."""

    return DecisionSample(
        game_id=game_id,
        ply_index=ply_index,
        slot=slot,
        action_id=action_id,
        setting=setting,
        enabled_action_count=policy.enabled_action_count,
        selected_probability=policy.selected_probability,
        selected_rank=policy.selected_rank,
        preferred_action_id=policy.preferred_action_id,
        preferred_probability=policy.preferred_probability,
    )


def _cell(
    setting: DecisionSetting | None,
    samples: Sequence[DecisionSample],
) -> DecisionCell:
    """Aggregate one group of samples."""

    followed = sum(1 for sample in samples if sample.followed_policy)
    departures = [sample for sample in samples if not sample.followed_policy]
    regrets = [sample.policy_regret for sample in samples]
    return DecisionCell(
        setting=setting,
        decisions=len(samples),
        followed_policy=followed,
        departures=len(departures),
        preferred_selection_rate=followed / len(samples),
        policy_regret=_mean(regrets),
        departure_policy_regret=_mean([sample.policy_regret for sample in departures]),
        maximum_policy_regret=max(regrets),
        preferred_probability=_mean(
            [sample.preferred_probability for sample in samples]
        ),
        selected_probability=_mean([sample.selected_probability for sample in samples]),
        selected_rank=_mean([float(sample.selected_rank) for sample in samples]),
        enabled_action_count=_mean(
            [float(sample.enabled_action_count) for sample in samples]
        ),
    )


def _seat_setting(seat: SeatRecord) -> DecisionSetting:
    """Read one seat's dials from the configuration it recorded."""

    configuration = seat.configuration
    return DecisionSetting(
        target_rating=_optional_int(configuration, "target_rating", seat.label),
        temperature=_optional_float(configuration, "temperature", seat.label),
    )


def _optional_int(
    configuration: Mapping[str, Any],
    key: str,
    label: str,
) -> int | None:
    value = configuration.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise DecisionDecompositionError(
            f"seat {label!r} recorded a non-integer {key}: {value!r}"
        )
    return value


def _optional_float(
    configuration: Mapping[str, Any],
    key: str,
    label: str,
) -> float | None:
    value = configuration.get(key)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise DecisionDecompositionError(
            f"seat {label!r} recorded a non-numeric {key}: {value!r}"
        )
    return float(value)


def _slot(turn: chess.Color) -> PlayerSlot:
    return "white" if turn == chess.WHITE else "black"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "DECISION_DECOMPOSITION_VERSION",
    "POLICY_DISTRIBUTION",
    "DecisionAnalysisConfig",
    "DecisionCell",
    "DecisionDecomposition",
    "DecisionDecompositionError",
    "DecisionSample",
    "DecisionSet",
    "DecisionSetting",
    "PlayedDecision",
    "collect_decisions",
    "decision_measurements",
    "decompose_game_records",
    "score_played_decisions",
    "summarize_decisions",
]
