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
    #: Why resampling the units a reading scored cannot estimate this metric's
    #: dispersion, when that follows from what the metric counts rather than
    #: from what has been characterized so far. Without it a report has one word
    #: for two situations: a sampling floor nobody has produced yet, and one
    #: that cannot exist. Only the first is worth waiting for.
    #:
    #: A metric that declares one is refused a floor from any estimator, not
    #: only a resampled one: the declaration says the quantity has no spread
    #: worth stating. It annotates the metric rather than redefining it, so it
    #: stays out of series identity and needs no ``definition_version`` bump.
    no_sampling_floor_reason: str | None = None


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
                        "no_sampling_floor_reason": metric.no_sampling_floor_reason,
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
            NormalizedColumn.CLOCK_REMAINING_DELTA_MS.value,
            NormalizedColumn.CLOCK_STATUS.value,
            NormalizedColumn.INITIAL_POSITION.value,
            NormalizedColumn.RULESET.value,
            NormalizedColumn.TIME_INCREMENT_MS.value,
            NormalizedColumn.TIME_INITIAL_MS.value,
            NormalizedColumn.WHITE_NORMALIZED_RATING.value,
        ),
    )
)

#: The held-out resignation reading consumes the same trajectory content move
#: prediction does, plus the derived termination columns. Those decide which
#: plies are positives: a game's terminal action is only in the action sequence
#: when the derivation attributed the ending to the side to move, so a
#: re-derivation that moved a game between categories or changed why a terminal
#: action was omitted changes what this metric was computed over.
TERMINATION_PREDICTION_PROJECTION = register_projection(
    DataProjection(
        name="termination_prediction",
        version=1,
        columns=(
            NormalizedColumn.ACTION_IDS.value,
            NormalizedColumn.BLACK_NORMALIZED_RATING.value,
            NormalizedColumn.INITIAL_POSITION.value,
            NormalizedColumn.RULESET.value,
            NormalizedColumn.TERMINAL_ACTION_STATUS.value,
            NormalizedColumn.TERMINATION_CATEGORY.value,
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

NOVELTY_FAMILY = register_family(
    MetricFamily(
        identifier="novelty",
        title="Novelty robustness",
        summary=(
            "What survives when the model is taken off distribution by a known "
            "dose of perturbation. Only measurements whose ground truth comes "
            "from the chess layer appear here, because human-referenced ones "
            "are undefined once a prefix stops being what the humans played."
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

#: Split from generated play rather than folded into it, because the two are
#: read on different terms. A generated-play metric describes the moves the
#: model played; this one describes what its policy says about stopping, read
#: on games humans played.
GAME_TERMINATION_FAMILY = register_family(
    MetricFamily(
        identifier="game-termination",
        title="Game termination",
        summary=(
            "Whether the policy wants to resign where a human did, and at the "
            "material a human would have. Read on frozen human games, against "
            "the human's own action rather than against a target rate."
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

#: Split from both efficiency families because it describes the instrument
#: rather than the product. Inference efficiency answers whether the engine is
#: fast enough to play against; this answers what reading it cost the harness,
#: which is a scope question about the suite. One family holding both would put
#: a benchmark that got slower and a model that got slower on the same shelf.
BENCHMARK_COST_FAMILY = register_family(
    MetricFamily(
        identifier="benchmark-cost",
        title="Benchmark cost",
        summary=(
            "What one benchmark invocation cost to take, on the machine that "
            "took it and under the configuration it was given. Read to decide "
            "what a sweep can afford rather than to judge a checkpoint."
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

#: Opening frequency tiers, stated here for the reason the phase names are: a
#: slice layer that renamed or re-bounded a tier would end these series, which
#: is intended, because a tier's share boundaries are what its number means.
OPENING_TIER_SLICE_NAMES: tuple[str, ...] = (
    "common_opening",
    "uncommon_opening",
    "rare_opening",
    "very_rare_opening",
    "unseen_opening",
    "unclassified_opening",
)

#: Held-out loss against how often the training selection saw each opening.
#: Reported only when the checkpoint trained on the corpus the pool was drawn
#: from, which is what keeps a tier's membership pinned by the same digest the
#: fingerprint already carries. See the rare-opening tail section of
#: `docs/evaluation.md`.
HELD_OUT_MOVE_LOSS_BY_OPENING_TIER: Mapping[str, MetricDefinition] = {
    tier: register_metric(
        MetricDefinition(
            identifier=f"held_out.move_loss_{tier}",
            family=HELD_OUT_PREDICTION_FAMILY.identifier,
            direction=MetricDirection.LOWER_IS_BETTER,
            definition_version=1,
            summary=(
                f"Raw-logit cross-entropy of the human move on held-out games "
                f"in the {tier.removesuffix('_opening').replace('_', ' ')} "
                "opening-frequency tier, which a family enters by its share of "
                "the checkpoint's training selection. Read the tiers together "
                "rather than one alone: rare openings are harder to predict "
                "whether or not they are undertrained, and only the shape "
                "across frequencies tells the two apart."
            ),
            cost=MetricCost.SINGLE_PASS,
            projection=MOVE_PREDICTION_PROJECTION.name,
        )
    )
    for tier in OPENING_TIER_SLICE_NAMES
}

ADJUDICATED_PREDICATE_NAMES: tuple[str, ...] = (
    "mate_available",
    "mate_threatened",
    "material_gain",
    "only_move",
    "stalemate_available",
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

ADJUDICATED_BEST_RANK: Mapping[str, MetricDefinition] = {
    predicate: _adjudicated_metric(
        predicate,
        "best_rank",
        direction=MetricDirection.LOWER_IS_BETTER,
        summary=(
            f"Mean legal-masked rank of the best action that handles a "
            f"{predicate.replace('_', ' ')} position. One means the model "
            "preferred it; a large rank distinguishes an absence from the near "
            "miss the policy mass alone cannot separate."
        ),
    )
    for predicate in ADJUDICATED_PREDICATE_NAMES
}


def _novelty_metric(
    identifier: str,
    *,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    """Register one novelty dose-response metric.

    Every metric here is generated rather than a view pass, and every one is
    execution-sensitive: the perturbation recipe, seed, onset, and dose are
    realized inputs to the value, so two doses are two series rather than two
    readings of one. That is decision 0020's rule applied to derived positions
    instead of to generated games.

    The source games stay the data component, which is what lets a perturbed
    arm and the control arm it is read against share evaluation inputs while
    sitting on different series.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=NOVELTY_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.GENERATED,
            projection=MOVE_PREDICTION_PROJECTION.name,
            execution_sensitive=True,
        )
    )


#: Band names are part of metric identity, so they are stated here rather than
#: imported. A slice layer that renamed a band or moved its floor would end
#: these series, which is the intended behavior.
MATERIAL_GAIN_BAND_NAMES: tuple[str, ...] = ("pawn", "minor", "rook_or_better")

NOVELTY_MASK_PENALTY = _novelty_metric(
    "novelty.mask_penalty",
    direction=MetricDirection.LOWER_IS_BETTER,
    summary=(
        "Negative log of raw legal mass at derived positions on the arm's own dose."
    ),
)

NOVELTY_MASK_PENALTY_DELTA = _novelty_metric(
    "novelty.mask_penalty_delta",
    direction=MetricDirection.LOWER_IS_BETTER,
    summary=(
        "Extra legality penalty this dose costs over the same checkpoint's "
        "unperturbed reading of the same plies. A difference rather than a "
        "ratio: both sides approach zero as a checkpoint trains, and the "
        "ratio of two vanishing numbers grows with checkpoint quality."
    ),
)

NOVELTY_REALIZED_DOSE = _novelty_metric(
    "novelty.realized_dose",
    direction=MetricDirection.INFORMATIONAL,
    summary=(
        "Share of the opponent plies past the onset that were actually replaced "
        "by a random legal move. The configured dose is a rate, so the realized "
        "one is what the arm was measured at."
    ),
)

NOVELTY_DERIVED_PLY_RETENTION = _novelty_metric(
    "novelty.derived_ply_retention",
    direction=MetricDirection.INFORMATIONAL,
    summary=(
        "Scored positions on this arm over those on the control arm. A "
        "perturbation can make the human's own next move illegal, which ends "
        "the derivation, so this reports how much of the source play survived "
        "and how selected the surviving positions are."
    ),
)

NOVELTY_MATERIAL_GAIN_POLICY_MASS: Mapping[str, MetricDefinition] = {
    band: _novelty_metric(
        f"novelty.material_gain_policy_mass_{band}",
        direction=MetricDirection.HIGHER_IS_BETTER,
        summary=(
            f"Raw policy mass on the actions that win material, where the most "
            f"the position wins is {band.replace('_', ' ')}. Mass rather than a "
            "selected rate because the failure this benchmark was opened for "
            "was a winning capture ranked sixth at 3.7% in a position humans "
            "never reach, beside the same win taken first elsewhere."
        ),
    )
    for band in MATERIAL_GAIN_BAND_NAMES
}

NOVELTY_MATERIAL_GAIN_SELECTED_RATE: Mapping[str, MetricDefinition] = {
    band: _novelty_metric(
        f"novelty.material_gain_selected_rate_{band}",
        direction=MetricDirection.HIGHER_IS_BETTER,
        summary=(
            f"Rate at which the legal greedy action wins the material available, "
            f"where the most the position wins is {band.replace('_', ' ')}. The "
            "coarser half of the pair above: it moves later and by less, and it "
            "is what a player meets."
        ),
    )
    for band in MATERIAL_GAIN_BAND_NAMES
}

NOVELTY_MATERIAL_GAIN_OPPORTUNITY_SHARE: Mapping[str, MetricDefinition] = {
    band: _novelty_metric(
        f"novelty.material_gain_opportunity_share_{band}",
        direction=MetricDirection.INFORMATIONAL,
        summary=(
            f"Share of this arm's scored positions that offer a "
            f"{band.replace('_', ' ')} material win. The mix the bands exist to "
            "hold fixed, reported so a reading that moved because the "
            "perturbation changed the mix is visible rather than inferred."
        ),
    )
    for band in MATERIAL_GAIN_BAND_NAMES
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
    be scheduled into a training cadence as though it were a cheap scoring pass,
    and none consumes a data projection: the decisions come from games the model
    played rather than from any pass over a corpus.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=DECISION_DECOMPOSITION_FAMILY.identifier,
            direction=MetricDirection.INFORMATIONAL,
            definition_version=1,
            summary=summary,
            cost=MetricCost.GENERATED,
            execution_sensitive=True,
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
        no_sampling_floor_reason=(
            "the rate counts rating slices rather than games, so resampling "
            "the games a reading scored estimates the dispersion of a "
            "different quantity"
        ),
    )
)

DEPENDENCY_RATING_CROSS_CONDITIONING_PENALTY = register_metric(
    MetricDefinition(
        identifier="dependency.rating_cross_conditioning_penalty",
        family=CORRECTNESS_FAMILY.identifier,
        direction=MetricDirection.HIGHER_IS_BETTER,
        definition_version=1,
        summary=(
            "Mean extra move loss a position pays when scored at a "
            "conditioning rating outside its own band, against its own band's. "
            "The graded form of the match rate above, which saturates at one "
            "on any checkpoint that has learned the ordering at all."
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
        no_sampling_floor_reason=(
            "each rating slice is split at its own median prefix strength, so "
            "a game's contribution is not fixed under resampling and there is "
            "no per-unit contribution to bootstrap"
        ),
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


def _inference_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
    definition_version: int = 1,
    *,
    cost: MetricCost = MetricCost.MEASURED_EXECUTION,
    execution_sensitive: bool = True,
    no_sampling_floor_reason: str | None = None,
) -> MetricDefinition:
    """Register one inference-efficiency metric.

    A counted quantity overrides the timing defaults: it reads the same on
    every machine, and a workload-scoped series would split one number across
    the devices it was counted beside.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=INFERENCE_EFFICIENCY_FAMILY.identifier,
            direction=direction,
            definition_version=definition_version,
            summary=summary,
            cost=cost,
            execution_sensitive=execution_sensitive,
            no_sampling_floor_reason=no_sampling_floor_reason,
        )
    )


#: Why a counted quantity carries no floor.
_COUNTED_NOT_SAMPLED = (
    "the value is counted from the loaded module rather than sampled, so it is "
    "identical in every process and there is nothing for a spread to be over"
)

INFERENCE_PARAMETERS = _inference_metric(
    "inference.parameters",
    MetricDirection.INFORMATIONAL,
    (
        "Parameters in the loaded checkpoint, frozen ones included. Counted "
        "rather than timed, so it carries no noise and needs no floor."
    ),
    cost=MetricCost.FREE,
    execution_sensitive=False,
    no_sampling_floor_reason=_COUNTED_NOT_SAMPLED,
)

INFERENCE_DECISION_GFLOPS = _inference_metric(
    "inference.decision_gflops",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Billions of floating-point operations one decision costs the model, "
        "counted from an instrumented forward pass. This is the arithmetic a "
        "model change alters, and it is counted rather than timed because a "
        "wall clock understates it wherever the device is launch or bandwidth "
        "bound."
    ),
    cost=MetricCost.FREE,
    execution_sensitive=False,
    no_sampling_floor_reason=_COUNTED_NOT_SAMPLED,
)

INFERENCE_PEAK_MEMORY_MB = _inference_metric(
    "inference.peak_memory_mb",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Peak device memory one forward pass allocates at the declared compute "
        "batch size, in megabytes. Bounds the batch a device can serve."
    ),
)

INFERENCE_DECISION_OVERHEAD_MS = _inference_metric(
    "inference.decision_overhead_ms",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Milliseconds one batched decision spends around the model: context "
        "assembly, batch construction, legal masking and sampling. The host "
        "copy is inside both timed windows and cancels, so it is not here. "
        "Does not amortize with batch size, so it is the floor under "
        "any decision, and it is where an encoding or action-vocabulary change "
        "lands rather than in the forward pass."
    ),
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
        "Whole batched decisions resolved per second at the declared batch "
        "size, taken from the median batch. Spans context collection, batch "
        "construction, the forward pass, legal masking, and sampling, which is "
        "the loop the generated benchmarks run. Separate from latency because "
        "batching trades one for the other, and reading a serving figure as an "
        "interactive one is the usual way that trade gets hidden."
    ),
    definition_version=2,
)

INFERENCE_FORWARD_THROUGHPUT = _inference_metric(
    "inference.forward_throughput_per_second",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Decisions per second through the model call alone, at the declared "
        "compute batch size, on a batch built once and re-run and taken from "
        "the median batch. Measured where the device stops being launch bound, "
        "which is the only batch size at which a wall clock separates two model "
        "sizes at all, and excluding the host copy and finite check a served "
        "decision pays, which are fixed costs that would dilute it by more the "
        "wider the batch."
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

GENERATED_PLAY_WAYPOINT_GAME_RATE = _generated_play_metric(
    "generated_play.waypoint_game_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Share of games whose deepest book match was a position several named "
        "openings still pass through. A depth behavior rather than a "
        "repertoire choice, which is why the repertoire distribution excludes "
        "these games and this rate reports them instead."
    ),
)

GENERATED_PLAY_MEAN_BOOK_PLY = _generated_play_metric(
    "generated_play.mean_book_ply",
    MetricDirection.INFORMATIONAL,
    (
        "Mean book depth reached, in book plies, over the games the book named "
        "at all. Book coordinates rather than game plies, so a transposition "
        "and the main move order read the same depth."
    ),
)

GENERATED_PLAY_MEAN_AVAILABLE_PLY = _generated_play_metric(
    "generated_play.mean_available_ply",
    MetricDirection.INFORMATIONAL,
    (
        "Mean depth the book still ran onward from where the game left it. "
        "Read beside the depth reached: alone it says how well analyzed the "
        "chosen lines are rather than how well the model knows them."
    ),
)

GENERATED_PLAY_MEAN_CONSUMED_FRACTION = _generated_play_metric(
    "generated_play.mean_consumed_fraction",
    MetricDirection.INFORMATIONAL,
    (
        "Mean share of the theory still available that the game consumed. The "
        "part of book depth that separates playing an offbeat line to its end "
        "from abandoning a deeply analyzed one early."
    ),
)


#: Quantities compared against matched human play. Kept as plain strings here
#: because the registry is the bottom of the import graph; the evaluation layer
#: owns the enum and asserts the two agree.
GENERATED_PLAY_COMPARED_QUANTITIES: tuple[str, ...] = (
    "game-length",
    "result",
    "repetition",
    "cycle",
    "repertoire",
    "waypoint",
    "book-depth",
    "book-available-depth",
    "book-consumed-fraction",
    "move-diversity",
    "termination",
)


def _curve_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    """Register one human-reference curve metric for generated play."""

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


def _quantity_slug(quantity: str) -> str:
    """Return the metric-identifier form of a compared quantity's name."""

    return quantity.replace("-", "_")


#: The distances that say whether generated play looks human. Unlike the raw
#: rollout scalars these have a direction: closer to matched human play is
#: better, which is the project's stated goal rather than an invented target.
#: Read them against the reference levels each comparison estimates for itself,
#: since two finite samples of one population never agree exactly.
GENERATED_PLAY_CONDITIONAL_DISTANCE: Mapping[str, MetricDefinition] = {
    quantity: _curve_metric(
        f"generated_play.{_quantity_slug(quantity)}_conditional_distance",
        MetricDirection.LOWER_IS_BETTER,
        (
            f"Mean distance between the generated and human {quantity} curves "
            "over the rating points the model supports. The rating-conditional "
            "reading: whether the model plays like humans of the rating it was "
            "told to play at."
        ),
    )
    for quantity in GENERATED_PLAY_COMPARED_QUANTITIES
}

GENERATED_PLAY_POOLED_DISTANCE: Mapping[str, MetricDefinition] = {
    quantity: _curve_metric(
        f"generated_play.{_quantity_slug(quantity)}_pooled_distance",
        MetricDirection.LOWER_IS_BETTER,
        (
            f"Distance between the rating-free generated and human {quantity} "
            "distributions. Separate from the conditional reading because a "
            "model can match humans on average while matching none of them at "
            "any particular rating."
        ),
    )
    for quantity in GENERATED_PLAY_COMPARED_QUANTITIES
}

#: Informational rather than directional, and deliberately so: a flat curve
#: only means something beside the human curve it is supposed to follow. A
#: model whose pooled distribution matches while this sits at the no-response
#: null has learned the average human and is ignoring its rating input, which
#: is the behavioral form of a rating-dependency test.
GENERATED_PLAY_RATING_VARIATION: Mapping[str, MetricDefinition] = {
    quantity: _curve_metric(
        f"generated_play.{_quantity_slug(quantity)}_rating_variation",
        MetricDirection.INFORMATIONAL,
        (
            f"How much the generated {quantity} curve moves across the rating "
            "range, as the mean distance of its points from its own average. "
            "Read against the level a model with no rating response would show."
        ),
    )
    for quantity in GENERATED_PLAY_COMPARED_QUANTITIES
}


def _termination_held_out_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    """Register one resignation metric read off games humans played.

    A deterministic pass over a fixed view rather than a rollout, so it is
    scoped by the content it scored instead of by a generation recipe. That is
    what makes this family cheap enough to read at a training cadence while
    the generated readings beside it are not.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=GAME_TERMINATION_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.SINGLE_PASS,
            projection=TERMINATION_PREDICTION_PROJECTION.name,
        )
    )


#: Cheap and free of rollouts: the policy already assigns terminal-action mass
#: at every scored position, so every reading here comes out of one pass over
#: frozen human games.
TERMINATION_RESIGNATION_MASS_AT_RESIGNATION = _termination_held_out_metric(
    "termination.resignation_mass_at_resignation",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Mean raw probability the policy assigns the resignation action at the "
        "ply where the human actually resigned. The human's own action is the "
        "target here, which is why this one has a direction at all."
    ),
)

TERMINATION_RESIGNATION_MASS_AT_MOVES = _termination_held_out_metric(
    "termination.resignation_mass_at_moves",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Mean raw probability the policy assigns the resignation action at "
        "plies where the human moved instead. The false-positive half: a model "
        "that resigns everywhere scores well on the other reading alone."
    ),
)

TERMINATION_RESIGNATION_MASS_SEPARATION = _termination_held_out_metric(
    "termination.resignation_mass_separation",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "How much more resignation mass the policy places where humans "
        "resigned than where they moved. Neither half means much alone: both "
        "rise together on a model that has simply learned the action exists."
    ),
)

#: The same pass, read against material rather than pooled over every ply. A
#: model that resigns as often as humans overall can still be resigning in the
#: wrong positions, and the mass readings above average exactly that away.
TERMINATION_RESIGNATION_CALIBRATION_ERROR = _termination_held_out_metric(
    "termination.resignation_calibration_error",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Expected gap between the resignation mass the policy assigns and the "
        "rate humans resigned at, over positions holding the same material "
        "deficit, weighted by how often a position like that comes up. Both "
        "sides are read at the same plies of the same games, so the position "
        "distribution is shared rather than each side's own."
    ),
)

TERMINATION_RESIGNATION_CALIBRATION_GAP = _termination_held_out_metric(
    "termination.resignation_calibration_gap",
    MetricDirection.INFORMATIONAL,
    (
        "The same comparison signed rather than absolute: how much more "
        "resignation mass the policy spends than humans did over the same "
        "plies. Resigning too readily and too reluctantly are different "
        "findings that an absolute error merges."
    ),
)

#: The guardrails, read off generated play because they are about what the
#: model did rather than what it predicted. Reported as their own metrics
#: rather than inferred from the termination distance beside them, because
#: their failure modes are not symmetric and a distributional distance averages
#: exactly that asymmetry away.
GENERATED_PLAY_PREMATURE_RESIGNATION_RATE = _generated_play_metric(
    "generated_play.premature_resignation_rate",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Share of the model's resignations taken from positions it was not "
        "behind in on material. The product-critical failure: worse than never "
        "resigning, and invisible to every other benchmark. Material balance is "
        "a heuristic proxy, so read it against the human rate beside it rather "
        "than as an absolute."
    ),
)

GENERATED_PLAY_PREMATURE_RESIGNATION_HUMAN_RATE = _generated_play_metric(
    "generated_play.premature_resignation_human_rate",
    MetricDirection.INFORMATIONAL,
    (
        "The same rate computed over human resignations in the reference. The "
        "proxy's noise lands on both sides identically, which is what makes the "
        "model rate readable despite the proxy admitting unsound positions."
    ),
)

GENERATED_PLAY_SILENT_TERMINAL_ACTIONS = _generated_play_metric(
    "generated_play.silent_terminal_actions",
    MetricDirection.LOWER_IS_BETTER,
    (
        "How many enabled terminal actions the suite never once selected. The "
        "opposite failure to premature resignation: an action the runtime "
        "offers and the model never uses is a capability that exists only on "
        "paper. Counted over enabled actions alone, so a disabled action is "
        "absent from the reading rather than silently passing it."
    ),
)

GENERATED_PLAY_UNTIMED_NON_TERMINATION_RATE = _generated_play_metric(
    "generated_play.untimed_non_termination_rate",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Share of generated games that reached a claimable dead position and "
        "still never ended. The failure the draw-claim action exists to "
        "prevent: with no clock to resolve it, a model that shuffles in a drawn "
        "position and declines to claim has no way to finish the game."
    ),
)


_INTERVAL_MEAN_NO_FLOOR_REASON = (
    "a replicate of this reading is a second training run, and the intervals "
    "inside one share a process, an allocator and a warm device, so resampling "
    "them measures within-run jitter rather than the run-to-run variation a "
    "delta faces"
)


def _training_efficiency_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
    no_sampling_floor_reason: str,
) -> MetricDefinition:
    """Register one training-efficiency metric.

    These share the inference family's cost and sensitivity: their value is the
    execution rather than a property of the data, so they carry no projection
    and their declared workload joins series identity.

    They part company over the spread, which is why the reason is required here
    rather than optional. An inference reading replicates itself by running the
    benchmark again in another process; the only replicate of a training
    reading is a second training run. See
    ``docs/decisions/0061-a-training-cost-reading-has-no-replicate-to-resample.md``.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=TRAINING_EFFICIENCY_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.MEASURED_EXECUTION,
            execution_sensitive=True,
            no_sampling_floor_reason=no_sampling_floor_reason,
        )
    )


TRAINING_ACTIVE_POSITIONS_PER_SECOND = _training_efficiency_metric(
    "training.active_positions_per_second",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Active non-padding positions trained on per second of steady-state "
        "training, with warmup and identified overhead excluded. Padding is "
        "left out of the numerator because a configuration that pads more is "
        "not learning faster."
    ),
    _INTERVAL_MEAN_NO_FLOOR_REASON,
)

TRAINING_STEP_SECONDS = _training_efficiency_metric(
    "training.step_seconds",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Mean steady-state seconds per optimizer step. Reported beside "
        "throughput because the two move apart whenever the effective batch "
        "or the padding fraction changes."
    ),
    _INTERVAL_MEAN_NO_FLOOR_REASON,
)

TRAINING_PROCESSED_POSITIONS = _training_efficiency_metric(
    "training.processed_positions",
    MetricDirection.INFORMATIONAL,
    (
        "Active positions the run trained on in total. This is how large the "
        "run was rather than how well it did, and it is the budget axis a "
        "quality-versus-time report reads."
    ),
    (
        "the count is how large the run was rather than a sample of anything, "
        "so the reading holds no units to resample and a replicate of it is a "
        "second training run"
    ),
)

TRAINING_SECONDS = _training_efficiency_metric(
    "training.training_seconds",
    MetricDirection.INFORMATIONAL,
    (
        "Wall-clock seconds spent in optimizer steps, identified overhead "
        "removed. Informational because a longer run is not a worse one; it is "
        "the other budget axis rather than a quantity to minimize."
    ),
    (
        "this is one run's total wall clock rather than a mean over units, so "
        "the reading holds no units to resample and a replicate of it is a "
        "second training run"
    ),
)

TRAINING_ACTIVE_POSITION_FRACTION = _training_efficiency_metric(
    "training.active_position_fraction",
    MetricDirection.INFORMATIONAL,
    (
        "Share of the padded batch extent that carried a position to learn "
        "from. Explains a throughput change that came from batch composition "
        "rather than from execution speed."
    ),
    (
        "the loader's draw is pinned by a configuration the reading records as "
        "a coordinate, so resampling this reading's units redraws nothing; what "
        "else moves the fraction is which of those batches the window spanned, "
        "which is not a redraw either"
    ),
)

TRAINING_OVERHEAD_FRACTION = _training_efficiency_metric(
    "training.overhead_fraction",
    MetricDirection.INFORMATIONAL,
    (
        "Share of the run's wall clock spent outside measured training: "
        "startup, checkpoint writes, cadence evaluation, final validation, and "
        "instrumentation. Informational because evaluating more often is a "
        "deliberate choice rather than a regression."
    ),
    (
        "this divides whole-run totals, several of which the run paid exactly "
        "once, so the reading holds no units to resample and a replicate of it "
        "is a second training run"
    ),
)

TRAINING_PEAK_DEVICE_MEMORY_BYTES = _training_efficiency_metric(
    "training.peak_device_memory_bytes",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Peak bytes reserved from the device driver during the run. Reserved "
        "rather than allocated, because cached free blocks still decide "
        "whether a larger configuration fits."
    ),
    (
        "a run reaches its high-water mark once, so the reading holds no units "
        "to resample and a replicate of it is a second training run"
    ),
)

BENCHMARK_WALL_CLOCK_SECONDS = register_metric(
    MetricDefinition(
        identifier="benchmark.wall_clock_seconds",
        family=BENCHMARK_COST_FAMILY.identifier,
        direction=MetricDirection.LOWER_IS_BETTER,
        definition_version=1,
        summary=(
            "Wall-clock seconds one benchmark invocation spent in process, from "
            "the driver's first statement to the moment it finished assembling "
            "what it recorded. Interpreter startup, configuration loading and "
            "the store append are outside it, so a sweep's total is the sum of "
            "its steps plus what the driver itself pays."
        ),
        cost=MetricCost.MEASURED_EXECUTION,
        execution_sensitive=True,
    )
)

TRAINING_COMPILED_GRAPHS = _training_efficiency_metric(
    "training.compiled_graphs",
    MetricDirection.INFORMATIONAL,
    (
        "Graphs the compiler built during the run. Zero when compilation is "
        "off. A handful means the shapes the loader presents settled into a "
        "few specializations; a count that tracks the step count means the run "
        "recompiled instead of reusing, which reads as a poor speedup rather "
        "than as the fixable mistake it is."
    ),
    (
        "a run compiles the graphs it compiles, so the reading holds no units "
        "to resample and a replicate of it is a second training run"
    ),
)

TRAINING_STEP_SYNC_COST_SECONDS = _training_efficiency_metric(
    "training.step_sync_cost_seconds",
    MetricDirection.INFORMATIONAL,
    (
        "Added seconds per optimizer step when per-micro-batch device "
        "synchronization is not deferred to the logging interval, measured by "
        "interleaving both arms through the same window."
    ),
    (
        "interleaving the arms cancels within-run drift from the difference "
        "rather than measuring it, so the intervals are the two arms rather "
        "than replicates of the difference, of which a run produces one"
    ),
)


#: The shallow repertoire read exactly rather than from played games. A policy
#: at a fixed position is one forward pass, so an opening-tree walk that keeps
#: every line above a threshold produces the distribution with no sampling
#: noise at all. Separate series from the sampled repertoire distances: they
#: measure the same behavior by different instruments, at different depths, and
#: merging them would hide which one moved.
GENERATED_PLAY_EXACT_REPERTOIRE_CONDITIONAL_DISTANCE = _curve_metric(
    "generated_play.exact_repertoire_conditional_distance",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Mean distance between the exactly computed shallow repertoire and the "
        "human one, over the ratings the walk was run at. No sampling noise on "
        "the model side, so a delta here is the policy moving."
    ),
)

GENERATED_PLAY_EXACT_REPERTOIRE_POOLED_DISTANCE = _curve_metric(
    "generated_play.exact_repertoire_pooled_distance",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Distance between the rating-free exact repertoire and the human one, "
        "each averaged over the rating grid."
    ),
)

#: Where the opening divergence accumulates, rather than how large it is. The
#: exact walk beside this already reports the size; a model that is unlike
#: humans from the first move and one that follows theory and then leaves it
#: reach the same size and are different faults. Interpolated between plies, so
#: it is continuous rather than quantized to the truncations it was read at.
GENERATED_PLAY_DIVERGENCE_HALF_DEPTH = _curve_metric(
    "generated_play.divergence_half_depth",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Book ply by which half of the opening divergence against human play "
        "has accumulated, over the null each depth was judged against. Later "
        "means the model stays human-like deeper into theory."
    ),
)

GENERATED_PLAY_EXACT_REPERTOIRE_PRUNED_MASS = _curve_metric(
    "generated_play.exact_repertoire_pruned_mass",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Probability mass the walk stopped expanding below its threshold. The "
        "error bound on the exact reading rather than a discarded remainder: "
        "no rearrangement of those lines can move a label's share by more."
    ),
)

GENERATED_PLAY_EXACT_REPERTOIRE_UNSETTLED_MASS = _curve_metric(
    "generated_play.exact_repertoire_unsettled_mass",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Pruned mass whose label the book had not yet committed, so continuing "
        "it could still move the distribution. The bound worth reading: mass "
        "pruned on a destination keeps the label it already has, and the "
        "assumption-free bound beside it saturates near one on any real policy."
    ),
)

GENERATED_PLAY_EXACT_REPERTOIRE_WAYPOINT_MASS = _curve_metric(
    "generated_play.exact_repertoire_waypoint_mass",
    MetricDirection.INFORMATIONAL,
    (
        "Probability mass the walk left on waypoints rather than on a chosen "
        "opening. The exact counterpart of the sampled waypoint rate, at the "
        "walk's own shallow depth."
    ),
)


def _ladder_metric(
    identifier: str,
    direction: MetricDirection,
    summary: str,
) -> MetricDefinition:
    """Register one self-play rating-ladder metric.

    Rating behavior rather than generated play, because the games are the
    instrument rather than the subject: what is reported is where the
    configured dial landed, not what the play looked like. Every value is
    estimated from games the benchmark played, so the declared workload — the
    grid, the reference temperature, and the generation recipe — is an input to
    the value and joins series identity.
    """

    return register_metric(
        MetricDefinition(
            identifier=identifier,
            family=RATING_BEHAVIOR_FAMILY.identifier,
            direction=direction,
            definition_version=1,
            summary=summary,
            cost=MetricCost.GENERATED,
            execution_sensitive=True,
        )
    )


#: Most of the ladder is informational, and deliberately so. A self-play ladder
#: is internally consistent by construction, so a fitted rating and a slope
#: describe the transfer function rather than grade it. Three quantities carry a
#: direction the project is willing to name: ordering, the distance from the
#: configured scale, and the share of games that reached a result at all.
LADDER_FITTED_RATING = _ladder_metric(
    "ladder.fitted_rating",
    MetricDirection.INFORMATIONAL,
    (
        "Empirical rating fitted for one seat of the ladder, on the joint "
        "internal scale anchored at the reference temperature. Informational: "
        "a seat configured low is meant to read low."
    ),
)

LADDER_SCORE_RATE = _ladder_metric(
    "ladder.score_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Points per game one seat scored across its matches, counting a draw "
        "as a half. The raw observation the fit reads."
    ),
)

LADDER_SCORED_GAME_RATE = _ladder_metric(
    "ladder.scored_game_rate",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Share of one seat's games that reached a result rather than the ply "
        "limit. The limit sits past the longest game in the corpus, so a seat "
        "playing the way its corpus does would reach it essentially never, and "
        "the share rising is the model learning to finish games."
    ),
)

LADDER_PREFERRED_SELECTION_RATE = _ladder_metric(
    "ladder.preferred_selection_rate",
    MetricDirection.INFORMATIONAL,
    (
        "Share of one seat's decisions where the draw took an action the model "
        "ranked first. Part of the error profile read beside strength: a "
        "temperature that preserves average score while moving this has changed "
        "the shape of the mistakes rather than their number."
    ),
)

LADDER_POLICY_REGRET = _ladder_metric(
    "ladder.policy_regret",
    MetricDirection.INFORMATIONAL,
    (
        "Untempered probability one seat's selections gave up, over all of its "
        "decisions. Moves with temperature by construction, so it describes the "
        "dial rather than grading the model."
    ),
)

LADDER_DEPARTURE_POLICY_REGRET = _ladder_metric(
    "ladder.departure_policy_regret",
    MetricDirection.INFORMATIONAL,
    (
        "The same regret over the decisions that departed from the model's "
        "preference. Reported apart from the pooled figure because a small "
        "pooled value can mean either rare departures or near ties."
    ),
)

LADDER_SELECTED_RANK = _ladder_metric(
    "ladder.selected_rank",
    MetricDirection.INFORMATIONAL,
    (
        "Mean untempered rank of the action one seat played. Rank and "
        "probability are both reported, since rank two can carry nearly the "
        "preferred action's probability or almost none of it."
    ),
)

LADDER_RATING_ORDER_ACCURACY = _ladder_metric(
    "ladder.rating_order_accuracy",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "Pairwise accuracy with which a higher configured rating produces a "
        "higher fitted rating at one temperature. Ties score a half."
    ),
)

LADDER_ADJACENT_RATING_ORDER_ACCURACY = _ladder_metric(
    "ladder.adjacent_rating_order_accuracy",
    MetricDirection.HIGHER_IS_BETTER,
    (
        "The same accuracy over adjacent configured ratings only. Harder than "
        "the pairwise figure and the one that localizes where the relationship "
        "degrades, since a distant pair can order correctly while every "
        "neighbouring pair is indistinguishable."
    ),
)

LADDER_RATING_ERROR = _ladder_metric(
    "ladder.rating_ladder_error",
    MetricDirection.LOWER_IS_BETTER,
    (
        "Mean absolute distance in rating points between fitted and configured "
        "rating at one temperature, after anchoring. A compressed ladder reads "
        "high here even when its ordering is perfect."
    ),
)

LADDER_FITTED_RATING_SLOPE = _ladder_metric(
    "ladder.fitted_rating_slope",
    MetricDirection.INFORMATIONAL,
    (
        "Least-squares slope of fitted against configured rating at one "
        "temperature. One is a reference shape rather than a target: a slope "
        "below it usually points at uneven rating coverage in the training "
        "data, whose expected response is better data rather than a mapping "
        "applied at the configuration boundary."
    ),
)

LADDER_FITTED_RATING_SPAN = _ladder_metric(
    "ladder.fitted_rating_span",
    MetricDirection.INFORMATIONAL,
    (
        "Fitted rating points between the strongest and weakest configured "
        "seat at one temperature. Near zero is the degenerate ladder, which is "
        "a reportable outcome rather than a failed measurement."
    ),
)

LADDER_TEMPERATURE_RESPONSE = _ladder_metric(
    "ladder.temperature_strength_response",
    MetricDirection.INFORMATIONAL,
    (
        "Fitted rating points gained per unit of sampling temperature, "
        "averaged over the configured ratings. Expected to be negative, since "
        "sampling a weak move loses more than sampling a strong one gains."
    ),
)

LADDER_ABLATED_TEMPERATURE_RESPONSE = _ladder_metric(
    "ladder.ablated_temperature_strength_response",
    MetricDirection.INFORMATIONAL,
    (
        "The same response measured with rating conditioning ablated, on the "
        "same joint scale. The control the conditioned response is read "
        "against."
    ),
)

LADDER_TEMPERATURE_RESPONSE_ATTENUATION = _ladder_metric(
    "ladder.temperature_response_attenuation",
    MetricDirection.INFORMATIONAL,
    (
        "Share of the ablated temperature response the conditioned arm avoids: "
        "one minus the ratio of the two. Positive means rating conditioning "
        "resists part of the drift. An observation about the model rather than "
        "a property the project promises, so it declares no direction."
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

TRAINING_HEALTH_CLIP_RATE = register_metric(
    MetricDefinition(
        identifier="training_health.clip_rate",
        family=TRAINING_HEALTH_FAMILY.identifier,
        direction=MetricDirection.INFORMATIONAL,
        definition_version=1,
        cost=MetricCost.FREE,
        summary=(
            "Share of the reported interval's steps whose gradient norm "
            "exceeded the configured ceiling. Near zero means clipping is "
            "insurance against a spike; sustained means it is a learning-rate "
            "cap under another name."
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
