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
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np
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
from anthro_chess.data.schema import (
    SCHEMA_VERSION,
    NormalizedColumn,
    SplitName,
    clock_remaining_ms,
    row_game_id,
)
from anthro_chess.evaluation.aggregation import (
    OPENING_TIER_DIMENSION,
    PHASE_DIMENSION,
    RATING_DIMENSION,
    RULE_CASE_DIMENSION,
    SliceAggregator,
    SliceMembership,
    SliceTable,
    position_memberships,
)
from anthro_chess.evaluation.dependency import PositionContext, PositionKey
from anthro_chess.evaluation.noise import GameTotals, MetricTotal
from anthro_chess.evaluation.opening_frequency import OpeningFrequency
from anthro_chess.evaluation.openings import (
    OpeningClassificationError,
    classify_action_ids,
)
from anthro_chess.evaluation.policy import PositionColumns, PositionPolicy
from anthro_chess.evaluation.results import (
    DataComponent,
    Measurement,
    measurement,
)
from anthro_chess.evaluation.results.metrics import (
    HELD_OUT_LEGAL_MOVE_LOSS,
    HELD_OUT_MOVE_LOSS,
    HELD_OUT_MOVE_LOSS_BY_OPENING_TIER,
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
    PositionLabels,
    PositionSlices,
    board_from_encoding,
    position_labels,
    position_slices,
)


class EvaluationLoaderConfig(ConfigModel):
    """Batching for evaluation, which never shuffles and never drops a game.

    Neither field is free to retune for speed. Every position is scored exactly
    once at any batch size, but the forward pass is not bitwise reproducible
    across batch shapes, so two readings taken at different batch sizes agree
    only to a few significant digits rather than exactly.
    """

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
    contexts: Mapping[PositionKey, PositionContext]
    _labels: dict[PositionKey, PositionLabels] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    #: Which slice of every dimension each encoded position falls in, derived
    #: where the labels were rather than where the scores are. Absent for a
    #: caller that did not supply labels, which is one that would have paid to
    #: derive them for positions it may not score.
    derived_memberships: Mapping[str, SliceMembership] | None = field(
        default=None,
        compare=False,
    )
    _opening_families: dict[int, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def position_count(self) -> int:
        """Return how many decisions one scoring pass covers."""

        return len(self.plies)

    def opening_family(self, game_id: int) -> str:
        """Return one scored game's opening family, classifying it once.

        Classification replays a game against the book, so it is derived on
        demand for the same reason the rule-sensitive labels are. A reading
        that asks about one game asks about all of them, so the first call
        classifies the whole selection rather than indexing the rows to look
        one up later.
        """

        if not self._opening_families:
            for row in self.rows:
                try:
                    label = classify_action_ids(
                        row[NormalizedColumn.ACTION_IDS],
                        initial_position=str(row[NormalizedColumn.INITIAL_POSITION]),
                    )
                except OpeningClassificationError as error:
                    raise ScoringError(str(error)) from error
                self._opening_families[row_game_id(row)] = label.family
        return self._opening_families[game_id]

    def labels(self, key: PositionKey) -> PositionLabels:
        """Return one position's rule-sensitive labels, deriving them once.

        A reading over a frozen pool hands the pool's derived labels to
        :func:`build_scoring_inputs` rather than paying for them again.
        Positions no artifact covers, such as a perturbed continuation, are
        resolved here.
        """

        labels = self._labels.get(key)
        if labels is None:
            labels = position_labels(board_from_encoding(self.plies[key].board))
            self._labels[key] = labels
        return labels


def build_scoring_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: SplitName,
    batch_size: int,
    length_bucket_width: int | None,
    identity_sha256: str,
    labels: Mapping[PositionKey, PositionLabels] | None = None,
    encodings: Mapping[int, Sequence[PlyEncoding]] | None = None,
) -> ScoringInputs:
    """Encode normalized rows once and derive the slices every reading needs.

    ``labels`` are the rule-sensitive labels of these positions, already
    derived. Left out, :meth:`ScoringInputs.labels` resolves them on demand.

    ``encodings`` is the same for the games themselves, keyed by game id, for a
    caller that encoded them across processes. Encoding is deterministic, so
    which of the two produced them does not reach the result.
    """

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: row_game_id(row),
    )
    examples: list[SequenceExample] = []
    plies: dict[PositionKey, PlyEncoding] = {}
    slices: dict[PositionKey, PositionSlices] = {}
    contexts: dict[PositionKey, PositionContext] = {}
    for row in ordered:
        game_id = row_game_id(row)
        supplied = None if encodings is None else encodings.get(game_id)
        encoded = tuple(supplied) if supplied else encode_game(encoding_input(row))
        examples.append(
            SequenceExample(
                shard_index=0,
                game_id=game_id,
                start_ply=encoded[0].ply_index,
                plies=encoded,
            )
        )
        for ply in encoded:
            key = (ply.game_id, ply.ply_index)
            plies[key] = ply
            derived = position_slices(ply, DEFAULT_RATING_BANDS)
            slices[key] = derived
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
    keys = tuple(plies)
    memberships = (
        None
        if labels is None
        else position_memberships(
            [slices[key] for key in keys],
            [labels[key].characteristics for key in keys],
        )
    )
    inputs = ScoringInputs(
        rows=tuple(ordered),
        dataset=dataset,
        loader_config=loader_config,
        plies=plies,
        slices=slices,
        contexts=contexts,
        derived_memberships=memberships,
    )
    if labels is not None:
        inputs._labels.update(labels)
    return inputs


def aggregate_positions(
    positions: Iterable[PositionPolicy],
    inputs: ScoringInputs,
    *,
    opening_frequency: OpeningFrequency | None = None,
) -> SliceTable:
    """Aggregate scored positions into every slice they belong to.

    ``opening_frequency`` adds both opening dimensions. They hang off it rather
    than off the scoring pass because classifying a game costs a replay, the
    per-family table means nothing without the frequency axis to read it
    against, and a pass over perturbed or generated move sequences would be
    labelling something the book never described.
    """

    aggregator = SliceAggregator()
    accumulate_positions(
        aggregator, positions, inputs, opening_frequency=opening_frequency
    )
    return aggregator.compute()


def accumulate_positions(
    aggregator: SliceAggregator,
    positions: Iterable[PositionPolicy],
    inputs: ScoringInputs,
    *,
    opening_frequency: OpeningFrequency | None = None,
) -> None:
    """Add scored positions to a running aggregation under their slice labels.

    Separate from :func:`aggregate_positions` because a reading that scores its
    pool a batch at a time holds one aggregator across every batch, and the
    inputs a position's labels come from live only as long as its batch does.
    """

    columns = PositionColumns.from_records(tuple(positions))
    if not len(columns):
        return
    aggregator.accumulate(
        columns,
        slice_memberships(columns, inputs, opening_frequency=opening_frequency),
    )


def slice_memberships(
    columns: PositionColumns,
    inputs: ScoringInputs,
    *,
    opening_frequency: OpeningFrequency | None = None,
) -> dict[str, SliceMembership]:
    """Return which slice of every dimension each scored position falls in."""

    keys = list(zip(columns.game_ids, columns.ply_indices, strict=True))
    if (
        opening_frequency is None
        and inputs.derived_memberships is not None
        and keys == list(inputs.plies)
    ):
        return dict(inputs.derived_memberships)
    families: list[str] | None = None
    tiers: list[str] | None = None
    if opening_frequency is not None:
        families = [inputs.opening_family(game_id) for game_id in columns.game_ids]
        tiers = [opening_frequency.tier(family) for family in families]
    return position_memberships(
        [inputs.slices[key] for key in keys],
        [inputs.labels(key).characteristics for key in keys],
        opening_families=families,
        opening_tiers=tiers,
    )


def per_game_totals(
    columns: PositionColumns,
    memberships: Mapping[str, SliceMembership],
) -> tuple[GameTotals, ...]:
    """Return what each scored game contributes to every reported metric.

    This is the input a bootstrap resamples. It reads the same slice tables the
    reported measurement is computed from, so a floor can never be computed
    from a differently defined quantity than the value it qualifies.
    """

    rows: dict[int, list[int]] = {}
    for offset, game_id in enumerate(columns.game_ids):
        rows.setdefault(game_id, []).append(offset)
    totals: list[GameTotals] = []
    for game_id, offsets in sorted(rows.items()):
        selected = np.array(offsets, dtype=np.int64)
        aggregator = SliceAggregator()
        aggregator.accumulate(
            columns.select(selected),
            {
                dimension: membership.select(selected)
                for dimension, membership in memberships.items()
            },
        )
        totals.append(
            GameTotals(
                game_id=game_id,
                metrics=_slice_metric_totals(aggregator.compute()),
            )
        )
    return tuple(totals)


#: The columns :func:`encoding_input` reads. Declared beside the reader rather
#: than in each benchmark, because a benchmark that projects its pool read has
#: to project to at least these and would otherwise drift from them silently.
SCORED_COLUMNS = (
    NormalizedColumn.SCHEMA_VERSION.value,
    NormalizedColumn.SOURCE_ID.value,
    NormalizedColumn.SOURCE_GAME_KEY.value,
    NormalizedColumn.RULESET.value,
    NormalizedColumn.INITIAL_POSITION.value,
    NormalizedColumn.ACTION_IDS.value,
    NormalizedColumn.WHITE_NORMALIZED_RATING.value,
    NormalizedColumn.BLACK_NORMALIZED_RATING.value,
    NormalizedColumn.TIME_INITIAL_MS.value,
    NormalizedColumn.TIME_INCREMENT_MS.value,
    NormalizedColumn.CLOCK_REMAINING_DELTA_MS.value,
)


def encoding_input(row: Mapping[str, Any]) -> GameEncodingInput:
    """Return the encoder input for one normalized row."""

    if row[NormalizedColumn.SCHEMA_VERSION] != SCHEMA_VERSION:
        raise ScoringError(
            f"normalized game uses schema version "
            f"{row[NormalizedColumn.SCHEMA_VERSION]}; expected {SCHEMA_VERSION}"
        )
    return GameEncodingInput(
        game_id=row_game_id(row),
        ruleset=str(row[NormalizedColumn.RULESET]),
        initial_position=str(row[NormalizedColumn.INITIAL_POSITION]),
        action_ids=tuple(row[NormalizedColumn.ACTION_IDS]),
        white_normalized_rating=row[NormalizedColumn.WHITE_NORMALIZED_RATING],
        black_normalized_rating=row[NormalizedColumn.BLACK_NORMALIZED_RATING],
        time_initial_ms=row[NormalizedColumn.TIME_INITIAL_MS],
        time_increment_ms=row[NormalizedColumn.TIME_INCREMENT_MS],
        clock_remaining_ms=clock_remaining_ms(row),
    )


def rows_identity_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: object = None,
) -> str:
    """Return the loader identity for one scored selection of games."""

    digest = sha256()
    digest.update(str(context).encode())
    for row in sorted(rows, key=lambda item: row_game_id(item)):
        digest.update(f"\n{row_game_id(row)}".encode())
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
    (OPENING_TIER_DIMENSION, HELD_OUT_MOVE_LOSS_BY_OPENING_TIER, "move_loss"),
)

#: Dimensions a scoring pass cannot label on its own. The opening tiers need a
#: family count over the training selection, which only the end-of-run reading
#: opts into, so a caller validating a declared metric list against
#: :func:`slice_metric_identifiers` rejects one rather than accepting a metric
#: that would then report nothing.
_EXTERNALLY_LABELLED_DIMENSIONS = frozenset({OPENING_TIER_DIMENSION})


def _slice_metric_totals(slices: SliceTable) -> dict[str, MetricTotal]:
    """Return every metric's summed contribution over one slice table.

    A mean and the count behind it recover the sum, which is what a resample
    has to add up. Reading them back off the same tables the measurement uses
    keeps one definition of each metric rather than two.
    """

    totals: dict[str, MetricTotal] = {}
    overall = slices.overall
    for definition, attribute in _OVERALL_METRICS:
        totals[definition.identifier] = MetricTotal(
            total=float(getattr(overall, attribute)) * overall.position_count,
            positions=overall.position_count,
        )
    for cutoff, definition in HELD_OUT_TOP_K_ACCURACY.items():
        totals[definition.identifier] = MetricTotal(
            total=overall.accuracy(cutoff) * overall.position_count,
            positions=overall.position_count,
        )
    for dimension, definitions, attribute in _SLICED_METRICS:
        for name, definition in definitions.items():
            summary = slices.slice_summary(dimension, name)
            if summary is None:
                continue
            totals[definition.identifier] = MetricTotal(
                total=float(getattr(summary, attribute)) * summary.position_count,
                positions=summary.position_count,
            )
    return totals


def slice_metric_identifiers() -> frozenset[str]:
    """Return every metric a scoring pass alone can produce."""

    identifiers = {definition.identifier for definition, _ in _OVERALL_METRICS}
    identifiers.update(
        definition.identifier for definition in HELD_OUT_TOP_K_ACCURACY.values()
    )
    for dimension, definitions, _ in _SLICED_METRICS:
        if dimension in _EXTERNALLY_LABELLED_DIMENSIONS:
            continue
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
    "accumulate_positions",
    "aggregate_positions",
    "build_scoring_inputs",
    "encoding_input",
    "per_game_totals",
    "rows_identity_sha256",
    "slice_measurements",
    "slice_metric_identifiers",
]
