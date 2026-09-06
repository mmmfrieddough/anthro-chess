"""Human-referenced evaluation of decisions exact chess logic resolves."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from anthro_chess.evaluation.aggregation import UNRATED_SLICE
from anthro_chess.evaluation.curves import (
    PairedRateObservation,
    PointReferenceComparison,
    ReferenceRateSupport,
)
from anthro_chess.evaluation.dependency import PositionKey
from anthro_chess.evaluation.noise import GameTotals, MetricTotal
from anthro_chess.evaluation.policy import ActionSetPolicy
from anthro_chess.evaluation.results import DataComponent, Measurement, measurement
from anthro_chess.evaluation.results.metrics import (
    ADJUDICATED_BEST_RANK,
    ADJUDICATED_HUMAN_GAP,
    ADJUDICATED_HUMAN_GAP_BY_RATING_BAND,
    ADJUDICATED_HUMAN_RATE,
    ADJUDICATED_POLICY_MASS,
    ADJUDICATED_SELECTED_RATE,
)
from anthro_chess.evaluation.scoring import ScoringInputs
from anthro_chess.evaluation.slices import (
    PREDICATE_REGISTRY,
    PositionPredicate,
    PredicateClass,
)

ADJUDICATION_VERSION = 1


@dataclass(frozen=True)
class AdjudicatedPosition:
    """One predicate opportunity scored for both the human and the model."""

    game_id: int
    ply_index: int
    predicate: PositionPredicate
    classification: PredicateClass
    rating_band: str
    human_success: bool
    model_success: bool
    policy_mass: float
    #: ``None`` when no legal action handles the predicate, which is a real
    #: state: a threatened mate nothing prevents offers nothing to rank.
    best_rank: int | None


@dataclass(frozen=True)
class PredicateReport:
    """One predicate's overall reading and rating-band drill-down."""

    predicate: PositionPredicate
    classification: PredicateClass
    overall: PointReferenceComparison
    rating_bands: Mapping[str, PointReferenceComparison]
    #: ``None`` when no opportunity offered a rankable action.
    mean_best_rank: float | None
    #: How many opportunities contributed a rank. Below ``overall.opportunities``
    #: exactly when some offered no successful action to rank.
    rankable_opportunities: int

    def as_record(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "mean_best_rank": self.mean_best_rank,
            "rankable_opportunities": self.rankable_opportunities,
            "overall": self.overall.as_record(),
            "rating_bands": {
                name: summary.as_record()
                for name, summary in sorted(self.rating_bands.items())
            },
        }


@dataclass(frozen=True)
class AdjudicationReport:
    """Every predicate realized by one deterministic evaluation view.

    ``positions`` is ``None`` where the reading did not retain them, which is
    the default: they are one record per realized opportunity, which the
    canonical pool measures in millions, and the summary below is what every
    reported metric is computed from. ``DetailConfig.per_position`` turns them
    back on for a session that wants to look at the decisions themselves.
    """

    predicates: Mapping[PositionPredicate, PredicateReport]
    #: Per-game shares of the quantities this reading can resample, which is
    #: what its own dispersion is estimated from.
    per_game_totals: tuple[GameTotals, ...]
    positions: tuple[AdjudicatedPosition, ...] | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "version": ADJUDICATION_VERSION,
            "predicates": {
                predicate.value: report.as_record()
                for predicate, report in sorted(
                    self.predicates.items(), key=lambda item: item[0].value
                )
            },
            "positions": (
                None
                if self.positions is None
                else [
                    {
                        "game_id": position.game_id,
                        "ply_index": position.ply_index,
                        "predicate": position.predicate.value,
                        "classification": position.classification.value,
                        "rating_band": position.rating_band,
                        "human_success": position.human_success,
                        "model_success": position.model_success,
                        "policy_mass": position.policy_mass,
                        "best_rank": position.best_rank,
                    }
                    for position in self.positions
                ]
            ),
        }

    def measurements(self, component: DataComponent) -> tuple[Measurement, ...]:
        """Return the overall readings and their band drill-down."""

        values: list[Measurement] = []
        for predicate, report in sorted(
            self.predicates.items(), key=lambda item: item[0].value
        ):
            name = predicate.value
            summary = report.overall
            for definitions, value in (
                (ADJUDICATED_HUMAN_RATE, summary.human_rate),
                (ADJUDICATED_SELECTED_RATE, summary.model_rate),
                (ADJUDICATED_POLICY_MASS, summary.model_probability_mass),
                (ADJUDICATED_HUMAN_GAP, summary.human_gap),
            ):
                values.append(
                    measurement(
                        definitions[name].identifier,
                        value,
                        data=component,
                        sample_size=summary.opportunities,
                    )
                )
            # Registered bands rather than realized ones. A run with moved
            # band boundaries realizes names nothing registers, so its
            # drill-down stays in the record instead of continuing these series.
            for band, definition in ADJUDICATED_HUMAN_GAP_BY_RATING_BAND[name].items():
                in_band = report.rating_bands.get(band)
                if in_band is None:
                    continue
                values.append(
                    measurement(
                        definition.identifier,
                        in_band.human_gap,
                        data=component,
                        sample_size=in_band.opportunities,
                    )
                )
            if report.mean_best_rank is not None:
                values.append(
                    measurement(
                        ADJUDICATED_BEST_RANK[name].identifier,
                        report.mean_best_rank,
                        data=component,
                        sample_size=report.rankable_opportunities,
                    )
                )
        return tuple(values)


class AdjudicationAccumulator:
    """Running support for every predicate a reading realizes.

    A reading scores its pool a batch at a time and folds each batch in here,
    so the opportunities themselves live for one batch rather than for the
    whole pass. Every reported quantity is a sum, a count, or a rate over them,
    which is what makes that possible.

    One precondition, which the batch plan satisfies by construction: **all of
    a game's opportunities arrive in one call.** The effective sample size
    counts a game's opportunities together, so a game split across two calls
    would be counted as two.
    """

    def __init__(self, *, retain_positions: bool = False) -> None:
        self._retain_positions = retain_positions
        self._positions: list[AdjudicatedPosition] = []
        self._overall: dict[PositionPredicate, ReferenceRateSupport] = defaultdict(
            ReferenceRateSupport
        )
        self._bands: dict[PositionPredicate, dict[str, ReferenceRateSupport]] = (
            defaultdict(lambda: defaultdict(ReferenceRateSupport))
        )
        self._rank_totals: dict[PositionPredicate, int] = defaultdict(int)
        self._rank_counts: dict[PositionPredicate, int] = defaultdict(int)
        self._game_totals: list[GameTotals] = []

    def add(self, scored: Sequence[ActionSetPolicy], inputs: ScoringInputs) -> None:
        """Fold one batch's scored predicate opportunities into the running support."""

        positions = adjudicated_positions(scored, inputs)
        if not positions:
            return
        if self._retain_positions:
            self._positions.extend(positions)

        by_game: dict[int, list[AdjudicatedPosition]] = defaultdict(list)
        by_predicate: dict[PositionPredicate, list[PairedRateObservation]] = (
            defaultdict(list)
        )
        by_band: dict[tuple[PositionPredicate, str], list[PairedRateObservation]] = (
            defaultdict(list)
        )
        for position in positions:
            by_game[position.game_id].append(position)
            observation = PairedRateObservation(
                game_id=position.game_id,
                human_success=position.human_success,
                model_success=position.model_success,
                model_probability_mass=position.policy_mass,
            )
            by_predicate[position.predicate].append(observation)
            by_band[(position.predicate, position.rating_band)].append(observation)
            if position.best_rank is not None:
                self._rank_totals[position.predicate] += position.best_rank
                self._rank_counts[position.predicate] += 1
        for game_id, group in sorted(by_game.items()):
            self._game_totals.append(_game_totals(game_id, group))
        # A game's opportunities for one predicate reach the overall support in
        # one call. Its two players can sit in different rating bands, so
        # adding band by band would count the game once per band it touched.
        for predicate, observations in by_predicate.items():
            self._overall[predicate].add(observations)
        for (predicate, band), observations in by_band.items():
            self._bands[predicate][band].add(observations)

    def report(self) -> AdjudicationReport | None:
        """Return everything the folded batches measured, or nothing realized."""

        if not self._overall:
            return None
        reports: dict[PositionPredicate, PredicateReport] = {}
        for predicate in PREDICATE_REGISTRY:
            support = self._overall.get(predicate)
            if support is None:
                continue
            count = self._rank_counts[predicate]
            reports[predicate] = PredicateReport(
                predicate=predicate,
                classification=PREDICATE_REGISTRY[predicate].classification,
                overall=support.summary(),
                rating_bands={
                    band: member.summary()
                    for band, member in sorted(self._bands[predicate].items())
                },
                mean_best_rank=(
                    self._rank_totals[predicate] / count if count else None
                ),
                rankable_opportunities=count,
            )
        return AdjudicationReport(
            predicates=reports,
            per_game_totals=tuple(
                sorted(self._game_totals, key=lambda total: total.game_id)
            ),
            positions=(
                tuple(
                    sorted(
                        self._positions,
                        key=lambda item: (
                            item.game_id,
                            item.ply_index,
                            item.predicate.value,
                        ),
                    )
                )
                if self._retain_positions
                else None
            ),
        )


def adjudicated_positions(
    scored: Sequence[ActionSetPolicy],
    inputs: ScoringInputs,
) -> tuple[AdjudicatedPosition, ...]:
    """Join action-set policy readings to exact predicates and human targets."""

    by_key = {(item.game_id, item.ply_index, item.name): item for item in scored}
    positions: list[AdjudicatedPosition] = []
    for key in inputs.plies:
        matches = inputs.labels(key).predicates
        ply = inputs.plies[key]
        rating_band = inputs.slices[key].rating_band or UNRATED_SLICE
        for predicate, match in matches.items():
            score = by_key.get((key[0], key[1], predicate.value))
            if score is None:
                raise ValueError(
                    f"predicate {predicate.value!r} at position {key} was not scored"
                )
            definition = PREDICATE_REGISTRY[predicate]
            positions.append(
                AdjudicatedPosition(
                    game_id=key[0],
                    ply_index=key[1],
                    predicate=predicate,
                    classification=definition.classification,
                    rating_band=rating_band,
                    human_success=(ply.target_action_id in match.successful_action_ids),
                    model_success=(
                        score.selected_action_id in match.successful_action_ids
                    ),
                    policy_mass=score.raw_probability_mass,
                    best_rank=score.best_rank,
                )
            )
    return tuple(positions)


def _game_totals(
    game_id: int,
    positions: Sequence[AdjudicatedPosition],
) -> GameTotals:
    """Return one game's clustered contribution for data-sampling floors."""

    metrics: dict[str, MetricTotal] = {}
    by_predicate: dict[PositionPredicate, list[AdjudicatedPosition]] = defaultdict(list)
    for position in positions:
        by_predicate[position.predicate].append(position)
    for predicate, group in by_predicate.items():
        name = predicate.value
        count = len(group)
        human = sum(float(item.human_success) for item in group)
        selected = sum(float(item.model_success) for item in group)
        mass = sum(item.policy_mass for item in group)
        metrics[ADJUDICATED_HUMAN_RATE[name].identifier] = MetricTotal(
            total=human,
            positions=count,
        )
        metrics[ADJUDICATED_SELECTED_RATE[name].identifier] = MetricTotal(
            total=selected,
            positions=count,
        )
        metrics[ADJUDICATED_POLICY_MASS[name].identifier] = MetricTotal(
            total=mass,
            positions=count,
        )
        metrics[ADJUDICATED_HUMAN_GAP[name].identifier] = MetricTotal(
            total=selected - human,
            positions=count,
        )
        ranks = [item.best_rank for item in group if item.best_rank is not None]
        if ranks:
            metrics[ADJUDICATED_BEST_RANK[name].identifier] = MetricTotal(
                total=float(sum(ranks)),
                positions=len(ranks),
            )
        by_band: dict[str, list[AdjudicatedPosition]] = defaultdict(list)
        for item in group:
            by_band[item.rating_band].append(item)
        for band, members in by_band.items():
            definition = ADJUDICATED_HUMAN_GAP_BY_RATING_BAND[name].get(band)
            if definition is None:
                continue
            metrics[definition.identifier] = MetricTotal(
                total=sum(
                    float(item.model_success) - float(item.human_success)
                    for item in members
                ),
                positions=len(members),
            )
    return GameTotals(game_id=game_id, metrics=metrics)


def action_sets(
    inputs: ScoringInputs,
    keys: Collection[PositionKey] | None = None,
) -> Mapping[PositionKey, Mapping[str, frozenset[int]]]:
    """Return the successful subsets the policy scorer consumes.

    ``keys`` narrows the result to the positions a reading keeps. A benchmark
    that scores a window inside longer games passes it, because resolving the
    predicates of a position it will discard is the whole cost of the position
    and buys nothing.
    """

    return {
        key: {
            predicate.value: match.successful_action_ids
            for predicate, match in matches.items()
        }
        for key in (inputs.plies if keys is None else keys)
        if (matches := inputs.labels(key).predicates)
    }


def merge_game_totals(
    *groups: Sequence[GameTotals],
) -> tuple[GameTotals, ...]:
    """Merge independent metric contributions under their shared game ids."""

    merged: dict[int, dict[str, MetricTotal]] = defaultdict(dict)
    for group in groups:
        for game in group:
            overlap = set(merged[game.game_id]).intersection(game.metrics)
            if overlap:
                raise ValueError(
                    f"game {game.game_id} reports duplicate metric totals: "
                    f"{', '.join(sorted(overlap))}"
                )
            merged[game.game_id].update(game.metrics)
    return tuple(
        GameTotals(game_id=game_id, metrics=metrics)
        for game_id, metrics in sorted(merged.items())
    )


__all__ = [
    "ADJUDICATION_VERSION",
    "AdjudicatedPosition",
    "AdjudicationAccumulator",
    "AdjudicationReport",
    "PredicateReport",
    "action_sets",
    "adjudicated_positions",
    "merge_game_totals",
]
