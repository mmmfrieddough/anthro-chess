"""Stable identity for every metric a benchmark can report.

Metric identity is a contract rather than an implementation detail. A metric
can be named in an issue, a report, or a chart without consulting a schema,
and no reader has to infer whether lower is better.

Changing what a metric means produces a new identity instead of silently
redefining an existing series, so ``definition_version`` is part of the
fingerprint. Bumping it ends the old series and starts a new one, which is
exactly the intent: a quietly redefined metric is the failure this module
exists to prevent.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from anthro_chess.data.schema import NormalizedColumn

METRIC_IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"


class MetricRegistryError(ValueError):
    """Raised when a metric, family, or projection is unknown or conflicting."""


class MetricDirection(StrEnum):
    """Which way a metric has to move to count as an improvement."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    #: The metric explains other movement rather than improving on its own.
    INFORMATIONAL = "informational"


#: Nominal passes charged for a repeated-pass metric. A dependency test scores
#: its view once per conditioning treatment, and the exact count depends on the
#: configured conditioning grid. Budgeting uses one nominal figure rather than
#: resolving the grid, because the point is to reject an unaffordable pairing
#: rather than to predict a runtime.
NOMINAL_REPEATED_PASSES = 8


class MetricCost(StrEnum):
    """What one reading of a metric costs, expressed in passes over its view.

    Cost is declared per metric so a schedule can reject an unaffordable
    pairing instead of silently slowing a training run. It is deliberately
    counted in view passes rather than seconds: the same schedule has to
    resolve identically on every machine.
    """

    #: Derivable from tensors a caller already computed. No view at all.
    FREE = "free"
    #: One scoring pass over the view.
    SINGLE_PASS = "single_pass"
    #: Several scoring passes over the view, one per conditioning treatment.
    REPEATED_PASS = "repeated_pass"
    #: Needs generated games rather than a pass over stored positions.
    GENERATED = "generated"
    #: Reads no view but costs real wall clock, because timing execution is
    #: the measurement. Distinct from ``FREE``: both pass over nothing, but a
    #: free metric is derived from work already done and this one is the work.
    MEASURED_EXECUTION = "measured_execution"

    @property
    def view_passes(self) -> int | None:
        """Return the passes one reading costs, or ``None`` when unbounded.

        A generated-play reading has no view to pass over, so its cost cannot
        be compared against a per-step position budget at all.
        """

        return _VIEW_PASSES[self]

    @property
    def reads_data(self) -> bool:
        """Return whether one reading consumes a projection of the pool.

        Generating is the one cost this cannot answer on its own: a generated
        reading may score decisions at pool positions or play whole games from
        the standard start. Such a metric declares which by naming a projection
        or not, so treat this as the default rather than the rule.
        """

        return self not in _NO_DATA_COSTS


@dataclass(frozen=True)
class DataProjection:
    """The input fields one measurement actually consumes.

    Most projections name normalized game columns. An explicitly external
    projection names fields in an owned benchmark dataset, such as the puzzle
    set. Fingerprints digest either form rather than the whole input record, so
    adding a field that no existing metric reads cannot break an unrelated
    series.
    """

    name: str
    version: int
    columns: tuple[str, ...]
    external: bool = False

    def as_record(self) -> dict[str, object]:
        """Return the stable record stored beside a content digest."""

        record: dict[str, object] = {
            "name": self.name,
            "version": self.version,
            "columns": list(self.columns),
        }
        if self.external:
            record["external"] = True
        return record


@dataclass(frozen=True)
class MetricFamily:
    """A group of metrics reported and read together."""

    identifier: str
    title: str
    summary: str


_VIEW_PASSES: Mapping[MetricCost, int | None] = {
    MetricCost.FREE: 0,
    MetricCost.SINGLE_PASS: 1,
    MetricCost.REPEATED_PASS: NOMINAL_REPEATED_PASSES,
    MetricCost.GENERATED: None,
    MetricCost.MEASURED_EXECUTION: 0,
}

#: Costs that consume no projection of the evaluation pool.
_NO_DATA_COSTS = frozenset({MetricCost.FREE, MetricCost.MEASURED_EXECUTION})

#: Costs that may consume a projection but need not. Generating is the only
#: one, because it covers two genuinely different readings: generating
#: decisions at positions a human reached, which does read pool content, and
#: playing whole games from the standard start, which reads none. A rollout's
#: human prefixes are an input to generation rather than the content the metric
#: was computed over, so a data component would misdescribe it, and the
#: declared generation recipe identifies the series instead.
_OPTIONAL_DATA_COSTS = frozenset({MetricCost.GENERATED})

#: Costs whose declared settings are realized inputs to the measured value, so
#: a metric declaring one may carry a workload component. Timing execution and
#: generating games are the two: a latency figure means nothing without the ply
#: depth it was taken at, and a rollout figure means nothing without the rating
#: and temperature it was played at. See
#: ``docs/decisions/0020-declared-settings-scope-generated-series.md``.
_WORKLOAD_SCOPED_COSTS = frozenset(
    {MetricCost.MEASURED_EXECUTION, MetricCost.GENERATED}
)


@dataclass(frozen=True)
class MetricDefinition:
    """One metric's durable identity."""

    identifier: str
    family: str
    direction: MetricDirection
    definition_version: int
    summary: str
    cost: MetricCost
    #: ``None`` for a metric with no data dependency, such as an optimizer or
    #: parameter statistic. Those carry a null data component in their
    #: fingerprint rather than a synthetic empty view, so they stay immune to
    #: every change in evaluation inputs.
    projection: str | None = None
    #: Whether the conditions the measurement ran under are inputs to its
    #: value. True for efficiency metrics, whose declared workload says what was
    #: timed, and for generated-play metrics, whose declared workload says what
    #: was played. It has two consequences: the declared workload joins series
    #: identity, because a different workload measures a different quantity; and
    #: the environment is recorded so a report can attribute a delta rather than
    #: credit it to the model.
    execution_sensitive: bool = False


_PROJECTIONS: dict[str, DataProjection] = {}
_FAMILIES: dict[str, MetricFamily] = {}
_METRICS: dict[str, MetricDefinition] = {}


def register_projection(projection: DataProjection) -> DataProjection:
    """Register a projection, rejecting a conflicting redefinition."""

    existing = _PROJECTIONS.get(projection.name)
    if existing is not None and existing != projection:
        raise MetricRegistryError(
            f"projection {projection.name!r} is already registered differently"
        )
    unknown = (
        ()
        if projection.external
        else tuple(
            column for column in projection.columns if column not in _NORMALIZED_COLUMNS
        )
    )
    if unknown:
        raise MetricRegistryError(
            f"projection {projection.name!r} names unknown normalized "
            f"column(s): {', '.join(unknown)}"
        )
    if tuple(sorted(set(projection.columns))) != projection.columns:
        raise MetricRegistryError(
            f"projection {projection.name!r} columns must be sorted and unique"
        )
    _PROJECTIONS[projection.name] = projection
    return projection


def register_family(family: MetricFamily) -> MetricFamily:
    """Register a metric family, rejecting a conflicting redefinition."""

    existing = _FAMILIES.get(family.identifier)
    if existing is not None and existing != family:
        raise MetricRegistryError(
            f"metric family {family.identifier!r} is already registered differently"
        )
    _FAMILIES[family.identifier] = family
    return family


def register_metric(metric: MetricDefinition) -> MetricDefinition:
    """Register a metric, rejecting a conflicting redefinition.

    A changed definition must arrive as a new ``definition_version`` rather
    than as an edit to a registered one, so this refuses to overwrite an
    identity that results already reference.
    """

    if metric.family not in _FAMILIES:
        raise MetricRegistryError(
            f"metric {metric.identifier!r} names unknown family {metric.family!r}"
        )
    if metric.projection is not None and metric.projection not in _PROJECTIONS:
        raise MetricRegistryError(
            f"metric {metric.identifier!r} names unknown projection "
            f"{metric.projection!r}"
        )
    if re.fullmatch(METRIC_IDENTIFIER_PATTERN, metric.identifier) is None:
        raise MetricRegistryError(
            f"metric identifier {metric.identifier!r} must be dotted lowercase"
        )
    if metric.definition_version < 1:
        raise MetricRegistryError(
            f"metric {metric.identifier!r} must declare a definition version "
            "of 1 or more"
        )
    if metric.cost not in _OPTIONAL_DATA_COSTS and metric.cost.reads_data != (
        metric.projection is not None
    ):
        raise MetricRegistryError(
            f"metric {metric.identifier!r} declares cost {metric.cost.value!r} "
            f"with projection {metric.projection!r}; a metric that passes over "
            "no view reads no data, and one that reads data has to name the "
            "projection it reads"
        )
    if metric.execution_sensitive and metric.cost not in _WORKLOAD_SCOPED_COSTS:
        raise MetricRegistryError(
            f"metric {metric.identifier!r} is execution-sensitive but declares "
            f"cost {metric.cost.value!r}; the declared workload is a realized "
            "input only to a metric that times its own execution or generates "
            "the games it reads"
        )
    existing = _METRICS.get(metric.identifier)
    if existing is not None and existing != metric:
        raise MetricRegistryError(
            f"metric {metric.identifier!r} is already registered differently; "
            "a changed definition needs a new identity"
        )
    _METRICS[metric.identifier] = metric
    return metric


def metric_definition(identifier: str) -> MetricDefinition:
    """Return one registered metric definition."""

    try:
        return _METRICS[identifier]
    except KeyError:
        raise MetricRegistryError(f"unknown metric: {identifier}") from None


def metric_family(identifier: str) -> MetricFamily:
    """Return one registered metric family."""

    try:
        return _FAMILIES[identifier]
    except KeyError:
        raise MetricRegistryError(f"unknown metric family: {identifier}") from None


def data_projection(name: str) -> DataProjection:
    """Return one registered data projection."""

    try:
        return _PROJECTIONS[name]
    except KeyError:
        raise MetricRegistryError(f"unknown data projection: {name}") from None


def registered_families() -> tuple[MetricFamily, ...]:
    """Return every registered family in identifier order."""

    return tuple(_FAMILIES[key] for key in sorted(_FAMILIES))


def registered_metrics(family: str | None = None) -> tuple[MetricDefinition, ...]:
    """Return registered metrics, optionally restricted to one family."""

    if family is not None and family not in _FAMILIES:
        raise MetricRegistryError(f"unknown metric family: {family}")
    return tuple(
        _METRICS[key]
        for key in sorted(_METRICS)
        if family is None or _METRICS[key].family == family
    )


def registered_projections() -> tuple[DataProjection, ...]:
    """Return every registered projection in name order."""

    return tuple(_PROJECTIONS[key] for key in sorted(_PROJECTIONS))


def iter_registry() -> Iterator[tuple[MetricFamily, tuple[MetricDefinition, ...]]]:
    """Iterate families with their metrics, for reports and documentation."""

    for family in registered_families():
        yield family, registered_metrics(family.identifier)


@dataclass(frozen=True)
class RegistrySnapshot:
    """A copy of the registry, taken before additional registration."""

    projections: tuple[DataProjection, ...]
    families: tuple[MetricFamily, ...]
    metrics: tuple[MetricDefinition, ...]


def registry_snapshot() -> RegistrySnapshot:
    """Capture the registry so temporary registrations can be undone.

    Registration is process-global by design, because a metric identity is a
    project-wide contract. A caller that adds definitions temporarily still
    needs a way back.
    """

    return RegistrySnapshot(
        projections=registered_projections(),
        families=registered_families(),
        metrics=registered_metrics(),
    )


def restore_registry(snapshot: RegistrySnapshot) -> None:
    """Restore a captured registry, discarding anything registered since."""

    _PROJECTIONS.clear()
    _FAMILIES.clear()
    _METRICS.clear()
    _PROJECTIONS.update(
        {projection.name: projection for projection in snapshot.projections}
    )
    _FAMILIES.update({family.identifier: family for family in snapshot.families})
    _METRICS.update({metric.identifier: metric for metric in snapshot.metrics})


def registry_record() -> dict[str, object]:
    """Return a machine-readable description of the whole registry."""

    return {
        "families": [
            {
                "identifier": family.identifier,
                "title": family.title,
                "summary": family.summary,
                "metrics": [
                    {
                        "identifier": metric.identifier,
                        "direction": metric.direction.value,
                        "definition_version": metric.definition_version,
                        "cost": metric.cost.value,
                        "projection": metric.projection,
                        "execution_sensitive": metric.execution_sensitive,
                        "summary": metric.summary,
                    }
                    for metric in metrics
                ],
            }
            for family, metrics in iter_registry()
        ],
        "projections": [
            projection.as_record() for projection in registered_projections()
        ],
    }


_NORMALIZED_COLUMNS: Mapping[str, None] = {
    column.value: None for column in NormalizedColumn
}


MOVE_PREDICTION_PROJECTION = register_projection(
    DataProjection(
        name="move_prediction",
        version=1,
        columns=(
            NormalizedColumn.ACTION_IDS.value,
            NormalizedColumn.BLACK_NORMALIZED_RATING.value,
            NormalizedColumn.INITIAL_POSITION.value,
            NormalizedColumn.RULESET.value,
            NormalizedColumn.WHITE_NORMALIZED_RATING.value,
        ),
    )
)

MOVE_TIMING_PROJECTION = register_projection(
    DataProjection(
        name="move_timing",
        version=1,
        columns=(
            NormalizedColumn.ACTION_IDS.value,
            NormalizedColumn.BLACK_NORMALIZED_RATING.value,
            NormalizedColumn.CLOCK_REMAINING_MS.value,
            NormalizedColumn.CLOCK_STATUS.value,
            NormalizedColumn.INITIAL_POSITION.value,
            NormalizedColumn.RULESET.value,
            NormalizedColumn.TIME_INCREMENT_MS.value,
            NormalizedColumn.TIME_INITIAL_MS.value,
            NormalizedColumn.WHITE_NORMALIZED_RATING.value,
        ),
    )
)

PUZZLE_RESPONSE_PROJECTION = register_projection(
    DataProjection(
        name="puzzle_response",
        version=1,
        columns=(
            "initial_fen",
            "moves",
            "puzzle_id",
            "rating",
            "source_game_key",
        ),
        external=True,
    )
)


TRAINING_HEALTH_FAMILY = register_family(
    MetricFamily(
        identifier="training-health",
        title="Training health",
        summary=(
            "Whether a run is going wrong right now. Measured on training "
            "batches or on the optimizer itself, so it never shares a series "
            "with a held-out quality metric."
        ),
    )
)

HELD_OUT_PREDICTION_FAMILY = register_family(
    MetricFamily(
        identifier="held-out-prediction",
        title="Held-out prediction",
        summary="How well the model predicts human moves it never trained on.",
    )
)

LEGALITY_FAMILY = register_family(
    MetricFamily(
        identifier="legality",
        title="Legality",
        summary=(
            "How much probability the raw model gives illegal moves before the "
            "runtime legal mask corrects it."
        ),
    )
)

ADJUDICATED_DECISIONS_FAMILY = register_family(
    MetricFamily(
        identifier="adjudicated-decisions",
        title="Adjudicated decisions",
        summary=(
            "How the model handles rare decisions whose immediate outcome exact "
            "chess logic resolves, read against the rate humans achieve in the "
            "same positions."
        ),
    )
)

CORRECTNESS_FAMILY = register_family(
    MetricFamily(
        identifier="correctness",
        title="Correctness",
        summary=(
            "Whether the pipeline learns what it was wired to learn. Dependency "
            "tests live here rather than with quality metrics: a weak result "
            "can mean an undertrained checkpoint rather than a defect, so these "
            "are read against training maturity."
        ),
    )
)

RATING_BEHAVIOR_FAMILY = register_family(
    MetricFamily(
        identifier="rating-behavior",
        title="Rating behavior",
        summary=(
            "Whether the configured target rating orders and calibrates the "
            "strength the model actually plays at."
        ),
    )
)

DECISION_DECOMPOSITION_FAMILY = register_family(
    MetricFamily(
        identifier="decision-decomposition",
        title="Decision decomposition",
        summary=(
            "Whether a bad decision came from the model preferring a bad action "
            "or from sampling drawing an action the model did not prefer. The "
            "two need opposite fixes and are indistinguishable in any aggregate "
            "that reports only how the game turned out."
        ),
    )
)

GENERATED_PLAY_FAMILY = register_family(
    MetricFamily(
        identifier="generated-play",
        title="Generated play",
        summary=(
            "What whole generated games look like, beyond what per-position "
            "prediction metrics can show."
        ),
    )
)

TIMING_FAMILY = register_family(
    MetricFamily(
        identifier="timing",
        title="Timing",
        summary=(
            "Whether move times match human timing behavior for the configured "
            "rating and clock context."
        ),
    )
)

#: Efficiency is split from training health because the two are read on
#: different terms. A training-health metric is a statement about the model
#: alone; an efficiency metric is a statement about the model on a machine
#: under a workload, so its delta needs an attribution before it means
#: anything. One family holding both would have no coherent answer to what a
#: movement in it implies.
TRAINING_EFFICIENCY_FAMILY = register_family(
    MetricFamily(
        identifier="training-efficiency",
        title="Training efficiency",
        summary=(
            "How fast and how cheaply a training configuration runs. Scoped to "
            "a run rather than a checkpoint, so it is measured during training "
            "and is not part of the end-of-run checkpoint suite."
        ),
    )
)

INFERENCE_EFFICIENCY_FAMILY = register_family(
    MetricFamily(
        identifier="inference-efficiency",
        title="Inference efficiency",
        summary=(
            "What a checkpoint costs to play with: move latency, throughput, "
            "and cold start. Scoped to a checkpoint rather than a run, and part "
            "of the suite, because an opponent too slow to play against is a "
            "product failure regardless of how it scores elsewhere."
        ),
    )
)


HELD_OUT_MOVE_LOSS = register_metric(
    MetricDefinition(
        identifier="held_out.move_loss",
        family=HELD_OUT_PREDICTION_FAMILY.identifier,
        direction=MetricDirection.LOWER_IS_BETTER,
        definition_version=1,
        summary="Raw-logit cross-entropy of the human move, before legal masking.",
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

HELD_OUT_LEGAL_MOVE_LOSS = register_metric(
    MetricDefinition(
        identifier="held_out.legal_move_loss",
        family=HELD_OUT_PREDICTION_FAMILY.identifier,
        direction=MetricDirection.LOWER_IS_BETTER,
        definition_version=1,
        summary="Cross-entropy of the human move after exact legal masking.",
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

HELD_OUT_UNIFORM_OVER_LEGAL_MOVE_LOSS = register_metric(
    MetricDefinition(
        identifier="held_out.uniform_over_legal_move_loss",
        family=HELD_OUT_PREDICTION_FAMILY.identifier,
        direction=MetricDirection.INFORMATIONAL,
        definition_version=1,
        summary=(
            "Cross-entropy of a uniform-over-legal policy on the same "
            "positions, which is the bar a model has to beat rather than a "
            "quantity to improve."
        ),
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

LEGALITY_MASK_PENALTY = register_metric(
    MetricDefinition(
        identifier="legality.mask_penalty",
        family=LEGALITY_FAMILY.identifier,
        direction=MetricDirection.LOWER_IS_BETTER,
        definition_version=1,
        summary="Negative log of the raw probability mass on legal moves.",
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

LEGALITY_LEGAL_MASS = register_metric(
    MetricDefinition(
        identifier="legality.legal_mass",
        family=LEGALITY_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary="Mean raw probability mass the model places on legal moves.",
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

LEGALITY_TOP1_ILLEGAL_RATE = register_metric(
    MetricDefinition(
        identifier="legality.top1_illegal_rate",
        family=LEGALITY_FAMILY.identifier,
        direction=MetricDirection.LOWER_IS_BETTER,
        definition_version=1,
        summary="Fraction of positions whose raw argmax action is illegal.",
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

HELD_OUT_TOP_K_ACCURACY: Mapping[int, MetricDefinition] = {
    cutoff: register_metric(
        MetricDefinition(
            identifier=f"held_out.top{cutoff}_accuracy",
            family=HELD_OUT_PREDICTION_FAMILY.identifier,
            direction=MetricDirection.HIGHER_IS_BETTER,
            definition_version=1,
            summary=(
                f"Fraction of held-out positions whose human move is in the "
                f"legal-masked top {cutoff}."
            ),
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    for cutoff in (1, 3, 5)
}

LEGALITY_TOP_ILLEGAL_FRACTION = register_metric(
    MetricDefinition(
        identifier="legality.top_illegal_fraction",
        family=LEGALITY_FAMILY.identifier,
        direction=MetricDirection.LOWER_IS_BETTER,
        definition_version=1,
        summary="Mean fraction of the raw top-5 actions that are illegal.",
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

LEGALITY_LEGAL_MARGIN = register_metric(
    MetricDefinition(
        identifier="legality.legal_margin",
        family=LEGALITY_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Mean gap between the best legal logit and the best illegal one. "
            "Says how close the raw model came to preferring an illegal move."
        ),
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

LEGALITY_LIFT = register_metric(
    MetricDefinition(
        identifier="legality.legality_lift",
        family=LEGALITY_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Mean log-odds of legal mass above uniform probability over the "
            "move vocabulary, which normalizes for how many moves are legal."
        ),
        cost=MetricCost.SINGLE_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)


#: Phase names are part of metric identity, so they are stated here rather
#: than imported. A slice layer that renamed a phase would end these series,
#: which is the intended behavior and is asserted by the tests.
PHASE_SLICE_NAMES: tuple[str, ...] = ("opening", "middlegame", "endgame")

#: The default rating bands. A run configured with different bands reports its
#: rating slices in the detail tier only, because a band boundary change means
#: a different measurement rather than a movement in an existing series.
RATING_BAND_SLICE_NAMES: tuple[str, ...] = (
    "under_1200",
    "1200_to_1599",
    "1600_to_1999",
    "2000_plus",
)

#: Slice name for positions whose player rating is unavailable.
UNRATED_SLICE_NAME = "unrated"

#: Rule cases that can hold at a position with a move to predict.
RULE_CASE_SLICE_NAMES: tuple[str, ...] = (
    "castling_available",
    "castling_rights",
    "check",
    "en_passant",
    "only_move",
    "pin",
    "promotion",
)


HELD_OUT_MOVE_LOSS_BY_PHASE: Mapping[str, MetricDefinition] = {
    phase: register_metric(
        MetricDefinition(
            identifier=f"held_out.move_loss_{phase}",
            family=HELD_OUT_PREDICTION_FAMILY.identifier,
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=1,
            summary=(
                f"Raw-logit cross-entropy of the human move on {phase} "
                "positions, held fixed so a shift in phase composition is not "
                "read as a change in prediction quality."
            ),
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    for phase in PHASE_SLICE_NAMES
}

HELD_OUT_MOVE_LOSS_BY_RATING_BAND: Mapping[str, MetricDefinition] = {
    band: register_metric(
        MetricDefinition(
            identifier=f"held_out.move_loss_{band}",
            family=HELD_OUT_PREDICTION_FAMILY.identifier,
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=1,
            summary=(
                f"Raw-logit cross-entropy of the human move on {band} "
                "positions. This measures how hard those positions are to "
                "predict, not whether the model reads the rating input."
            ),
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    for band in (*RATING_BAND_SLICE_NAMES, UNRATED_SLICE_NAME)
}

LEGALITY_MASK_PENALTY_BY_PHASE: Mapping[str, MetricDefinition] = {
    phase: register_metric(
        MetricDefinition(
            identifier=f"legality.mask_penalty_{phase}",
            family=LEGALITY_FAMILY.identifier,
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=1,
            summary=(
                f"Negative log of raw legal mass on {phase} positions. Legality "
                "varies severalfold across phases, so the pool-wide mean sits "
                "between populations rather than describing any of them."
            ),
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    for phase in PHASE_SLICE_NAMES
}

LEGALITY_MASK_PENALTY_BY_RULE_CASE: Mapping[str, MetricDefinition] = {
    case: register_metric(
        MetricDefinition(
            identifier=f"legality.mask_penalty_{case}",
            family=LEGALITY_FAMILY.identifier,
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=1,
            summary=(
                f"Negative log of raw legal mass on held-out positions where "
                f"{case.replace('_', ' ')} applies. Rare rule cases vanish from "
                "a pool-wide average, which is what this slice prevents."
            ),
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    for case in RULE_CASE_SLICE_NAMES
}

ADJUDICATED_PREDICATE_NAMES: tuple[str, ...] = (
    "mate_available",
    "mate_threatened",
    "only_move",
    "stalemate_available",
    "stalemate_reply",
)


def _adjudicated_metric(
    predicate: str,
    suffix: str,
    *,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    return register_metric(
        MetricDefinition(
            identifier=f"adjudicated.{predicate}_{suffix}",
            family=ADJUDICATED_DECISIONS_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )


ADJUDICATED_HUMAN_RATE: Mapping[str, MetricDefinition] = {
    predicate: _adjudicated_metric(
        predicate,
        "human_rate",
        direction=MetricDirection.INFORMATIONAL,
        summary=(
            f"Rate at which held-out humans handled {predicate.replace('_', ' ')} "
            "positions, the reference for the model rather than a perfect-play target."
        ),
    )
    for predicate in ADJUDICATED_PREDICATE_NAMES
}

ADJUDICATED_SELECTED_RATE: Mapping[str, MetricDefinition] = {
    predicate: _adjudicated_metric(
        predicate,
        "selected_rate",
        direction=MetricDirection.INFORMATIONAL,
        summary=(
            f"Rate at which the model's legal greedy action handles "
            f"{predicate.replace('_', ' ')} positions."
        ),
    )
    for predicate in ADJUDICATED_PREDICATE_NAMES
}

ADJUDICATED_POLICY_MASS: Mapping[str, MetricDefinition] = {
    predicate: _adjudicated_metric(
        predicate,
        "policy_mass",
        direction=MetricDirection.INFORMATIONAL,
        summary=(
            f"Raw policy mass assigned to actions that handle "
            f"{predicate.replace('_', ' ')} positions."
        ),
    )
    for predicate in ADJUDICATED_PREDICATE_NAMES
}

ADJUDICATED_HUMAN_GAP: Mapping[str, MetricDefinition] = {
    predicate: _adjudicated_metric(
        predicate,
        "human_gap",
        direction=MetricDirection.INFORMATIONAL,
        summary=(
            f"Model selected-action rate minus the held-out human rate for "
            f"{predicate.replace('_', ' ')} positions. Zero is a match; the sign "
            "shows whether the model over- or under-converts."
        ),
    )
    for predicate in ADJUDICATED_PREDICATE_NAMES
}


def _decision_metric(identifier: str, summary: str) -> MetricDefinition:
    """Register one decision-decomposition metric.

    Every metric in this family is informational, and deliberately so. Each one
    moves with temperature by construction: a run at temperature zero selects
    the preferred action every time and gives up no probability, which is not an
    improvement but a different tradeoff between the two error classes. A
    declared direction here would invite a report to reward turning the dial
    down.

    They are also generated-play readings rather than view passes, so none can
    be scheduled into a training cadence as though it were a cheap scoring pass.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=DECISION_DECOMPOSITION_FAMILY.identifier,
            direction=MetricDirection.INFORMATIONAL,
            definition_version=1,
            summary=summary,
            cost=MetricCost.GENERATED,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )


DECISION_PREFERRED_SELECTION_RATE = _decision_metric(
    "decision.preferred_selection_rate",
    (
        "Fraction of decisions whose selected action was the model's own "
        "preferred action. The complement is the share of decisions sampling "
        "decided rather than the model."
    ),
)

DECISION_POLICY_REGRET = _decision_metric(
    "decision.policy_regret",
    (
        "Mean untempered policy probability given up by the selected action "
        "against the preferred one, over every decision. Zero means the draw "
        "never overrode the model."
    ),
)

DECISION_DEPARTURE_POLICY_REGRET = _decision_metric(
    "decision.departure_policy_regret",
    (
        "Mean probability given up over the decisions that departed from the "
        "preferred action alone. Small values mean the draws that overrode the "
        "model were near ties rather than rejections of a confident preference, "
        "which the pooled figure cannot distinguish from rare large losses."
    ),
)

DECISION_PREFERRED_PROBABILITY = _decision_metric(
    "decision.preferred_probability",
    (
        "Mean untempered probability the model gave its preferred action. This "
        "is the policy's own sharpness, measured independently of the "
        "temperature the draw used."
    ),
)

DECISION_SELECTED_RANK = _decision_metric(
    "decision.selected_rank",
    (
        "Mean rank of the selected action in the untempered policy. Reported "
        "beside the probabilities because rank and probability disagree: rank "
        "two can carry nearly the preferred action's probability, or almost "
        "none of it."
    ),
)


DEPENDENCY_RATING_SHUFFLED_DEGRADATION = register_metric(
    MetricDefinition(
        identifier="dependency.rating_shuffled_degradation",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Increase in held-out move loss when each position's rating is "
            "replaced by another position's. Near zero means the model is not "
            "reading the input, or has not learned to yet."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

DEPENDENCY_RATING_CONSTANT_DEGRADATION = register_metric(
    MetricDefinition(
        identifier="dependency.rating_constant_degradation",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Increase in held-out move loss when every position is scored at "
            "one fixed rating."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

DEPENDENCY_RATING_ABSENT_DEGRADATION = register_metric(
    MetricDefinition(
        identifier="dependency.rating_absent_degradation",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Increase in held-out move loss when the rating input is marked "
            "absent on positions that have one."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

DEPENDENCY_RATING_CROSS_CONDITIONING_MATCH_RATE = register_metric(
    MetricDefinition(
        identifier="dependency.rating_cross_conditioning_match_rate",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Fraction of rating slices whose best conditioning value is the "
            "matching one. Sensitivity alone cannot establish direction; this "
            "can."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

DEPENDENCY_RATING_WITHIN_GAME_RESPONSE = register_metric(
    MetricDefinition(
        identifier="dependency.rating_within_game_response",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.INFORMATIONAL,
        definition_version=1,
        summary=(
            "How much further the policy leans toward strong-rating play when "
            "the prefix looked strong, at a fixed stated rating. Near zero "
            "means rating is treated as a static prior; both outcomes are "
            "useful to know rather than better or worse."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

DEPENDENCY_RATING_ANCHOR_POLICY_DIVERGENCE = register_metric(
    MetricDefinition(
        identifier="dependency.rating_anchor_policy_divergence",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Mean divergence between the legal-masked policies at the lowest "
            "and highest conditioning ratings. Says whether the dial moves the "
            "distribution the runtime actually samples from."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)

DEPENDENCY_RATING_ANCHOR_TOP1_AGREEMENT = register_metric(
    MetricDefinition(
        identifier="dependency.rating_anchor_top1_agreement",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.INFORMATIONAL,
        definition_version=1,
        summary=(
            "Fraction of positions whose greedy move is unchanged between the "
            "lowest and highest conditioning ratings. Explains how much of a "
            "policy shift survives discrete action selection."
        ),
        cost=MetricCost.REPEATED_PASS,
        projection=MOVE_PREDICTION_PROJECTION.name,
    )
)


def _puzzle_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=RATING_BEHAVIOR_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.REPEATED_PASS,
            projection=PUZZLE_RESPONSE_PROJECTION.name,
        )
    )


PUZZLE_GREEDY_FIRST_MOVE_ACCURACY = _puzzle_metric(
    "puzzle.greedy_first_move_accuracy",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Fraction of puzzles whose first player move is the legal-masked "
        "argmax, averaged across the declared configured-rating grid."
    ),
)

PUZZLE_GREEDY_LINE_COMPLETION = _puzzle_metric(
    "puzzle.greedy_line_completion",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Fraction of complete verified puzzle lines for which every player "
        "move is the legal-masked argmax, averaged across configured ratings."
    ),
)

PUZZLE_SAMPLED_FIRST_MOVE_SOLVE_RATE = _puzzle_metric(
    "puzzle.sampled_first_move_solve_rate",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Expected first-move solve rate under the declared reference "
        "temperature, averaged across the configured-rating grid."
    ),
)

PUZZLE_SAMPLED_LINE_COMPLETION = _puzzle_metric(
    "puzzle.sampled_line_completion",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Expected probability of sampling every player move in the verified "
        "line at the declared reference temperature."
    ),
)

PUZZLE_GREEDY_RATING_SLOPE = _puzzle_metric(
    "puzzle.greedy_rating_slope",
    MetricDirection.INFORMATIONAL,
    (
        "Linear slope of fitted puzzle rating against configured game rating "
        "under greedy selection. One is a reference shape, not a promotion gate."
    ),
)

PUZZLE_SAMPLED_RATING_SLOPE = _puzzle_metric(
    "puzzle.sampled_rating_slope",
    MetricDirection.INFORMATIONAL,
    (
        "Linear slope of fitted puzzle rating against configured game rating "
        "at the declared reference temperature."
    ),
)

PUZZLE_GREEDY_RATING_ORDER_ACCURACY = _puzzle_metric(
    "puzzle.greedy_rating_order_accuracy",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Pairwise accuracy with which higher configured ratings produce higher "
        "fitted puzzle ratings under greedy selection."
    ),
)

PUZZLE_SAMPLED_RATING_ORDER_ACCURACY = _puzzle_metric(
    "puzzle.sampled_rating_order_accuracy",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Pairwise accuracy with which higher configured ratings produce higher "
        "fitted puzzle ratings at the declared reference temperature."
    ),
)

PUZZLE_GREEDY_CURVE_DISTANCE = _puzzle_metric(
    "puzzle.greedy_curve_distance",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Mean absolute distance between the continuous human expected-solve "
        "curve and greedy full-line completion over puzzle rating."
    ),
)

PUZZLE_SAMPLED_CURVE_DISTANCE = _puzzle_metric(
    "puzzle.sampled_curve_distance",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Mean absolute distance between the continuous human expected-solve "
        "curve and sampled full-line completion over puzzle rating."
    ),
)

PUZZLE_TRAINING_OVERLAP_RATE = _puzzle_metric(
    "puzzle.training_overlap_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Fraction of puzzle source games present in the selected checkpoint's "
        "training or validation corpus. This is provenance, not model quality."
    ),
)


def _inference_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    """Register one inference-efficiency metric.

    Every metric in this family shares a cost and a sensitivity, so declaring
    them once keeps a later addition from quietly landing as a machine-blind
    series.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=INFERENCE_EFFICIENCY_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.MEASURED_EXECUTION,
            execution_sensitive=True,
        )
    )


#: Percentiles reported for batch-one move latency. The median says what play
#: usually feels like and the tail says whether it ever stalls; a mean alone
#: hides the second behind the first.
LATENCY_PERCENTILES: tuple[int, ...] = (50, 90, 99)

INFERENCE_MOVE_LATENCY_BY_PERCENTILE: Mapping[int, MetricDefinition] = {
    percentile: _inference_metric(
        f"inference.move_latency_p{percentile}_ms",
        MetricDirection.LOWER_IS_BETTER,
        (
            f"The p{percentile} wall-clock milliseconds one batch-one move "
            "decision takes end to end, spanning encoding, model execution, "
            "legal masking, and sampling."
        ),
    )
    for percentile in LATENCY_PERCENTILES
}

INFERENCE_MOVE_LATENCY_MEAN = _inference_metric(
    "inference.move_latency_mean_ms",
    MetricDirection.INFORMATIONAL,
    (
        "Mean milliseconds per batch-one move decision. Reported to relate "
        "latency to serving capacity rather than to be improved on its own, "
        "since the percentiles are what a player experiences."
    ),
)

INFERENCE_BATCH_THROUGHPUT = _inference_metric(
    "inference.batch_throughput_per_second",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Decisions resolved per second at the declared batch size. Separate "
        "from latency because batching trades one for the other, and reading a "
        "serving figure as an interactive one is the usual way that trade "
        "gets hidden."
    ),
)

INFERENCE_MODEL_LOAD_SECONDS = _inference_metric(
    "inference.model_load_seconds",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Seconds to resolve, validate, and load a checkpoint onto its device. "
        "Paid once per process and excluded from steady-state latency."
    ),
)

INFERENCE_FIRST_DECISION_SECONDS = _inference_metric(
    "inference.first_decision_seconds",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Seconds the first decision after loading takes. Lazy kernel "
        "compilation and allocator warmup land here rather than inflating the "
        "steady-state percentiles, which is why warmup is excluded there and "
        "measured here."
    ),
)


def _generated_play_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    """Register one generated-play metric.

    Every metric here is estimated from games the benchmark played, so they
    share a cost and a workload sensitivity. Declaring them once keeps a later
    addition from quietly landing as a series that ignores the rating and
    temperature it was played at.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=GENERATED_PLAY_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.GENERATED,
            execution_sensitive=True,
        )
    )


#: Most of this family is informational by construction. Human games are the
#: reference for whether generated play looks human, so a direction here would
#: assert a target the project has deliberately declined to hardcode: fewer
#: draws is not better, and neither is a longer game. The two exceptions are
#: defects rather than behaviors.
GENERATED_PLAY_MEAN_GAME_PLIES = _generated_play_metric(
    "generated_play.mean_game_plies",
    MetricDirection.INFORMATIONAL,
    "Mean moves per generated game, counting any replayed prefix.",
)

GENERATED_PLAY_MEAN_GENERATED_PLIES = _generated_play_metric(
    "generated_play.mean_generated_plies",
    MetricDirection.INFORMATIONAL,
    (
        "Mean moves a seat actually chose per game. Separate from total length "
        "because a prefix continuation's total is mostly the prefix."
    ),
)

GENERATED_PLAY_WHITE_SCORE = _generated_play_metric(
    "generated_play.white_score",
    MetricDirection.INFORMATIONAL,
    (
        "White's score per game, counting a draw as a half. Under self-play "
        "this is the color reading: one configuration in both seats should "
        "land near the human white score rather than at one extreme."
    ),
)

GENERATED_PLAY_DECISIVE_GAME_RATE = _generated_play_metric(
    "generated_play.decisive_game_rate",
    MetricDirection.INFORMATIONAL,
    "Share of finished games that ended with a winner rather than a draw.",
)

GENERATED_PLAY_UNFINISHED_GAME_RATE = _generated_play_metric(
    "generated_play.unfinished_game_rate",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Share of games the ply limit stopped before the rules ended them. "
        "Unlike the rest of this family this is a defect rather than a "
        "behavior: an unfinished game has no result or termination to compare "
        "against anything, so a high rate is the benchmark failing to observe "
        "endings at all."
    ),
)

GENERATED_PLAY_RESIGNATION_RATE = _generated_play_metric(
    "generated_play.resignation_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Share of games a seat ended by choosing the resignation action. Zero "
        "by construction when the runtime has resignation disabled."
    ),
)

GENERATED_PLAY_REPEATED_POSITION_GAME_RATE = _generated_play_metric(
    "generated_play.repeated_position_game_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Share of games that reached any position more than once. Reported for "
        "every game rather than only for repetition draws, because a model "
        "that shuffles and then moves on never terminates by repetition."
    ),
)

GENERATED_PLAY_THREEFOLD_CLAIMABLE_GAME_RATE = _generated_play_metric(
    "generated_play.threefold_claimable_game_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Share of games in which a threefold draw became claimable, whether or "
        "not the harness claimed it."
    ),
)

GENERATED_PLAY_MEAN_FIRST_REPETITION_PLY = _generated_play_metric(
    "generated_play.mean_first_repetition_ply",
    MetricDirection.INFORMATIONAL,
    (
        "Mean ply at which recurrence first appeared, averaged over the games "
        "that repeated at all. Says when a cycle started rather than how often "
        "one did, which is the rate above."
    ),
)

GENERATED_PLAY_MEAN_CYCLE_PLY_FRACTION = _generated_play_metric(
    "generated_play.mean_cycle_ply_fraction",
    MetricDirection.INFORMATIONAL,
    (
        "Share of the plies from first recurrence onward that were themselves "
        "recurrences, averaged over the games that repeated. Separates a game "
        "that touched a position twice and moved on from one that stayed in a "
        "cycle."
    ),
)

GENERATED_PLAY_MEAN_DISTINCT_MOVE_FRACTION = _generated_play_metric(
    "generated_play.mean_distinct_move_fraction",
    MetricDirection.INFORMATIONAL,
    (
        "Distinct moves as a share of moves played, averaged over games. Moves "
        "before either side reaches a repetition threshold."
    ),
)

GENERATED_PLAY_DISTINCT_GAME_FRACTION = _generated_play_metric(
    "generated_play.distinct_game_fraction",
    MetricDirection.INFORMATIONAL,
    (
        "Distinct move sequences as a share of games. A suite whose games are "
        "one trajectory reads one here, which is the signature of a collapsed "
        "policy — or simply of temperature zero, which is why this is read "
        "beside the temperature its series is scoped to rather than maximized."
    ),
)


TRAINING_HEALTH_GRADIENT_NORM = register_metric(
    MetricDefinition(
        identifier="training_health.gradient_norm",
        family=TRAINING_HEALTH_FAMILY.identifier,
        direction=MetricDirection.INFORMATIONAL,
        definition_version=1,
        cost=MetricCost.FREE,
        summary=(
            "Global gradient norm at the reported step. Distinguishes "
            "divergence and exploding gradients from a dead learning rate, "
            "which loss alone does not."
        ),
    )
)

TRAINING_HEALTH_UPDATE_TO_WEIGHT_RATIO = register_metric(
    MetricDefinition(
        identifier="training_health.update_to_weight_ratio",
        family=TRAINING_HEALTH_FAMILY.identifier,
        direction=MetricDirection.INFORMATIONAL,
        definition_version=1,
        cost=MetricCost.FREE,
        summary=(
            "Ratio of the optimizer update norm to the parameter norm at the "
            "reported step."
        ),
    )
)
