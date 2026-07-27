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

    @property
    def view_passes(self) -> int | None:
        """Return the passes one reading costs, or ``None`` when unbounded.

        A generated-play reading has no view to pass over, so its cost cannot
        be compared against a per-step position budget at all.
        """

        return _VIEW_PASSES[self]


@dataclass(frozen=True)
class DataProjection:
    """The normalized columns one measurement actually consumes.

    Fingerprints digest this projection rather than the whole normalized row,
    so adding a schema field that no existing metric reads cannot break an
    unrelated series. Timing fields and per-ply opening metadata are both
    expected; without a projection each would otherwise invalidate every
    series in the project on arrival.
    """

    name: str
    version: int
    columns: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return the stable record stored beside a content digest."""

        return {
            "name": self.name,
            "version": self.version,
            "columns": list(self.columns),
        }


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
}


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
    unknown = tuple(
        column for column in projection.columns if column not in _NORMALIZED_COLUMNS
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
    if (metric.cost is MetricCost.FREE) != (metric.projection is None):
        raise MetricRegistryError(
            f"metric {metric.identifier!r} declares cost {metric.cost.value!r} "
            f"with projection {metric.projection!r}; a free metric reads no "
            "data and a metric that reads data is never free"
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

#: Efficiency is split from training health because the two invalidate on
#: opposite terms. Training-health metrics carry no data component and are
#: immune to evaluation-input changes; efficiency metrics are dominated by an
#: environment component and are invalidated by a change of machine rather than
#: a change of model. One family holding both would have no coherent answer to
#: whether a series is still valid.
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
