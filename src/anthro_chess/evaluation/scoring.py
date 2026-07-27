"""The one path from normalized games to registered held-out measurements.

The end-of-run checkpoint suite and an in-training preview measure the same
quantities over different amounts of data. That only stays true if they share
an implementation, so encoding, slicing, aggregation, and the mapping from a
slice table onto registered metric identities all live here rather than inside
either caller.

Nothing in this module knows where its rows came from. A frozen test pool and
the validation split of a training corpus both arrive as normalized rows.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from pydantic import Field

from anthro_chess.config import ConfigModel
from anthro_chess.data import (
    GameEncodingInput,
    PlyEncoding,
    SequenceDataset,
    SequenceExample,
    SequenceLoaderConfig,
    encode_game,
)
from anthro_chess.data.schema import SCHEMA_VERSION, NormalizedColumn, SplitName
from anthro_chess.evaluation.aggregation import (
    PHASE_DIMENSION,
    RATING_DIMENSION,
    RULE_CASE_DIMENSION,
    SliceAggregator,
    SliceTable,
)
from anthro_chess.evaluation.dependency import PositionContext, PositionKey
from anthro_chess.evaluation.policy import PositionPolicy
from anthro_chess.evaluation.results import (
    DataComponent,
    Measurement,
    measurement,
)
from anthro_chess.evaluation.results.metrics import (
    HELD_OUT_LEGAL_MOVE_LOSS,
    HELD_OUT_MOVE_LOSS,
    HELD_OUT_MOVE_LOSS_BY_PHASE,
    HELD_OUT_MOVE_LOSS_BY_RATING_BAND,
    HELD_OUT_TOP_K_ACCURACY,
    HELD_OUT_UNIFORM_OVER_LEGAL_MOVE_LOSS,
    LEGALITY_LEGAL_MARGIN,
    LEGALITY_LEGAL_MASS,
    LEGALITY_LIFT,
    LEGALITY_MASK_PENALTY,
    LEGALITY_MASK_PENALTY_BY_PHASE,
    LEGALITY_MASK_PENALTY_BY_RULE_CASE,
    LEGALITY_TOP1_ILLEGAL_RATE,
    LEGALITY_TOP_ILLEGAL_FRACTION,
    MetricDefinition,
)
from anthro_chess.evaluation.slices import (
    DEFAULT_RATING_BANDS,
    PositionCharacteristic,
    PositionSlices,
    ply_characteristics,
    position_slices,
)


class EvaluationLoaderConfig(ConfigModel):
    """Batching for evaluation, which never shuffles and never drops a game."""

    batch_size: int = Field(default=8, ge=1)
    length_bucket_width: int | None = Field(default=32, ge=1)


class ScoringError(ValueError):
    """Raised when normalized rows cannot be prepared for scoring."""


@dataclass(frozen=True)
class ScoringInputs:
    """Encoded positions plus everything derived from them exactly once."""

    rows: tuple[dict[str, Any], ...]
    dataset: SequenceDataset
    loader_config: SequenceLoaderConfig
    plies: Mapping[PositionKey, PlyEncoding]
    slices: Mapping[PositionKey, PositionSlices]
    characteristics: Mapping[PositionKey, frozenset[PositionCharacteristic]]
    contexts: Mapping[PositionKey, PositionContext]

    @property
    def position_count(self) -> int:
        """Return how many decisions one scoring pass covers."""

        return len(self.plies)


def build_scoring_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: SplitName,
    batch_size: int,
    length_bucket_width: int | None,
    identity_sha256: str,
) -> ScoringInputs:
    """Encode normalized rows once and derive every per-position label."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: int(row[NormalizedColumn.GAME_ID]),
    )
    examples: list[SequenceExample] = []
    plies: dict[PositionKey, PlyEncoding] = {}
    slices: dict[PositionKey, PositionSlices] = {}
    characteristics: dict[PositionKey, frozenset[PositionCharacteristic]] = {}
    contexts: dict[PositionKey, PositionContext] = {}
    for row in ordered:
        encoded = encode_game(encoding_input(row))
        examples.append(
            SequenceExample(
                shard_index=0,
                game_id=int(row[NormalizedColumn.GAME_ID]),
                start_ply=encoded[0].ply_index,
                plies=encoded,
            )
        )
        for ply in encoded:
            key = (ply.game_id, ply.ply_index)
            plies[key] = ply
            derived = position_slices(ply, DEFAULT_RATING_BANDS)
            slices[key] = derived
            characteristics[key] = ply_characteristics(ply)
            contexts[key] = PositionContext(
                game_id=ply.game_id,
                ply_index=ply.ply_index,
                color=str(derived.color),
                rating=ply.target_rating,
                rating_band=derived.rating_band,
            )

    dataset = SequenceDataset(
        examples,
        identity_sha256=identity_sha256,
        split=split,
        chunk_length=None,
    )
    loader_config = SequenceLoaderConfig(
        split=split,
        batch_size=batch_size,
        length_bucket_width=length_bucket_width,
        chunk_length=None,
        shuffle=False,
        drop_last=False,
    )
    return ScoringInputs(
        rows=tuple(ordered),
        dataset=dataset,
        loader_config=loader_config,
        plies=plies,
        slices=slices,
        characteristics=characteristics,
        contexts=contexts,
    )


def aggregate_positions(
    positions: Iterable[PositionPolicy],
    inputs: ScoringInputs,
) -> SliceTable:
    """Aggregate scored positions into every slice they belong to."""

    aggregator = SliceAggregator()
    for position in positions:
        key = (position.game_id, position.ply_index)
        aggregator.add(position, inputs.slices[key], inputs.characteristics[key])
    return aggregator.compute()


def encoding_input(row: Mapping[str, Any]) -> GameEncodingInput:
    """Return the encoder input for one normalized row."""

    if row[NormalizedColumn.SCHEMA_VERSION] != SCHEMA_VERSION:
        raise ScoringError(
            f"normalized game uses schema version "
            f"{row[NormalizedColumn.SCHEMA_VERSION]}; expected {SCHEMA_VERSION}"
        )
    return GameEncodingInput(
        game_id=int(row[NormalizedColumn.GAME_ID]),
        ruleset=str(row[NormalizedColumn.RULESET]),
        initial_position=str(row[NormalizedColumn.INITIAL_POSITION]),
        action_ids=tuple(row[NormalizedColumn.ACTION_IDS]),
        white_normalized_rating=row[NormalizedColumn.WHITE_NORMALIZED_RATING],
        black_normalized_rating=row[NormalizedColumn.BLACK_NORMALIZED_RATING],
        time_initial_ms=row[NormalizedColumn.TIME_INITIAL_MS],
        time_increment_ms=row[NormalizedColumn.TIME_INCREMENT_MS],
        clock_remaining_ms=tuple(row[NormalizedColumn.CLOCK_REMAINING_MS]),
    )


def rows_identity_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: object = None,
) -> str:
    """Return the loader identity for one scored selection of games."""

    digest = sha256()
    digest.update(str(context).encode())
    for row in sorted(rows, key=lambda item: int(item[NormalizedColumn.GAME_ID])):
        digest.update(f"\n{row[NormalizedColumn.GAME_ID]}".encode())
    return digest.hexdigest()


#: Every overall metric a slice table can report, in the order it is measured.
_OVERALL_METRICS: tuple[tuple[MetricDefinition, str], ...] = (
    (HELD_OUT_MOVE_LOSS, "move_loss"),
    (HELD_OUT_LEGAL_MOVE_LOSS, "legal_move_loss"),
    (HELD_OUT_UNIFORM_OVER_LEGAL_MOVE_LOSS, "uniform_over_legal_move_loss"),
    (LEGALITY_MASK_PENALTY, "mask_penalty"),
    (LEGALITY_LEGAL_MASS, "legal_mass"),
    (LEGALITY_TOP1_ILLEGAL_RATE, "top1_illegal_rate"),
    (LEGALITY_TOP_ILLEGAL_FRACTION, "top_illegal_fraction"),
    (LEGALITY_LEGAL_MARGIN, "legal_margin"),
    (LEGALITY_LIFT, "legality_lift"),
)

#: Every sliced metric, as the dimension it slices and the summary field it
#: reads. A slice with no positions is absent rather than zero, so a metric
#: whose slice was never realized is simply not reported.
_SLICED_METRICS: tuple[tuple[str, Mapping[str, MetricDefinition], str], ...] = (
    (PHASE_DIMENSION, HELD_OUT_MOVE_LOSS_BY_PHASE, "move_loss"),
    (PHASE_DIMENSION, LEGALITY_MASK_PENALTY_BY_PHASE, "mask_penalty"),
    (RATING_DIMENSION, HELD_OUT_MOVE_LOSS_BY_RATING_BAND, "move_loss"),
    (RULE_CASE_DIMENSION, LEGALITY_MASK_PENALTY_BY_RULE_CASE, "mask_penalty"),
)


def slice_metric_identifiers() -> frozenset[str]:
    """Return every metric identifier a slice table can produce."""

    identifiers = {definition.identifier for definition, _ in _OVERALL_METRICS}
    identifiers.update(
        definition.identifier for definition in HELD_OUT_TOP_K_ACCURACY.values()
    )
    for _, definitions, _ in _SLICED_METRICS:
        identifiers.update(definition.identifier for definition in definitions.values())
    return frozenset(identifiers)


def slice_measurements(
    slices: SliceTable,
    component: DataComponent,
    *,
    metrics: Collection[str] | None = None,
) -> tuple[Measurement, ...]:
    """Return the registered measurements one slice table supports.

    ``metrics`` restricts the result to a declared subset, which is what a
    training cadence names. Passing ``None`` reports everything, which is what
    the canonical end-of-run reading does.
    """

    wanted = None if metrics is None else frozenset(metrics)
    overall = slices.overall
    values: list[Measurement] = []

    def include(definition: MetricDefinition) -> bool:
        return wanted is None or definition.identifier in wanted

    for definition, attribute in _OVERALL_METRICS:
        if include(definition):
            values.append(
                measurement(
                    definition.identifier,
                    float(getattr(overall, attribute)),
                    data=component,
                    sample_size=overall.position_count,
                )
            )
    for cutoff, definition in HELD_OUT_TOP_K_ACCURACY.items():
        if include(definition):
            values.append(
                measurement(
                    definition.identifier,
                    overall.accuracy(cutoff),
                    data=component,
                    sample_size=overall.position_count,
                )
            )
    for dimension, definitions, attribute in _SLICED_METRICS:
        for name, definition in definitions.items():
            if not include(definition):
                continue
            summary = slices.slice_summary(dimension, name)
            if summary is None:
                continue
            values.append(
                measurement(
                    definition.identifier,
                    float(getattr(summary, attribute)),
                    data=component,
                    sample_size=summary.position_count,
                )
            )
    return tuple(values)


__all__ = [
    "EvaluationLoaderConfig",
    "ScoringError",
    "ScoringInputs",
    "aggregate_positions",
    "build_scoring_inputs",
    "encoding_input",
    "rows_identity_sha256",
    "slice_measurements",
    "slice_metric_identifiers",
]
