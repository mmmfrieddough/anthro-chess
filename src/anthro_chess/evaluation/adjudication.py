"""Human-referenced evaluation of decisions exact chess logic resolves."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from anthro_chess.evaluation.aggregation import UNRATED_SLICE
from anthro_chess.evaluation.curves import (
    PairedRateObservation,
    PointReferenceComparison,
    compare_reference_rate,
)
from anthro_chess.evaluation.noise import GameTotals, MetricTotal
from anthro_chess.evaluation.policy import ActionSetPolicy
from anthro_chess.evaluation.results import DataComponent, Measurement, measurement
from anthro_chess.evaluation.results.metrics import (
    ADJUDICATED_BEST_RANK,
    ADJUDICATED_HUMAN_GAP,
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
    """Every predicate realized by one deterministic evaluation view."""

    positions: tuple[AdjudicatedPosition, ...]
    predicates: Mapping[PositionPredicate, PredicateReport]

    def as_record(self) -> dict[str, object]:
        return {
            "version": ADJUDICATION_VERSION,
            "predicates": {
                predicate.value: report.as_record()
                for predicate, report in sorted(
                    self.predicates.items(), key=lambda item: item[0].value
                )
            },
            "positions": [
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
            ],
        }

    def measurements(self, component: DataComponent) -> tuple[Measurement, ...]:
        """Return the bounded overall readings for the committed tier."""

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

    def per_game_totals(self) -> tuple[GameTotals, ...]:
        """Return game-clustered contributions for data-sampling floors."""

        grouped: dict[int, list[AdjudicatedPosition]] = defaultdict(list)
        for position in self.positions:
            grouped[position.game_id].append(position)

        totals: list[GameTotals] = []
        for game_id, positions in sorted(grouped.items()):
            metrics: dict[str, MetricTotal] = {}
            by_predicate: dict[PositionPredicate, list[AdjudicatedPosition]] = (
                defaultdict(list)
            )
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
            totals.append(GameTotals(game_id=game_id, metrics=metrics))
        return tuple(totals)


def build_adjudication_report(
    scored: Sequence[ActionSetPolicy],
    inputs: ScoringInputs,
) -> AdjudicationReport | None:
    """Join action-set policy readings to exact predicates and human targets."""

    by_key = {(item.game_id, item.ply_index, item.name): item for item in scored}
    positions: list[AdjudicatedPosition] = []
    for key, matches in inputs.predicates.items():
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
    if not positions:
        return None

    reports: dict[PositionPredicate, PredicateReport] = {}
    for predicate in PREDICATE_REGISTRY:
        matching = [item for item in positions if item.predicate is predicate]
        if not matching:
            continue
        bands = {
            band: _compare(item for item in matching if item.rating_band == band)
            for band in sorted({item.rating_band for item in matching})
        }
        ranks = [item.best_rank for item in matching if item.best_rank is not None]
        reports[predicate] = PredicateReport(
            predicate=predicate,
            classification=PREDICATE_REGISTRY[predicate].classification,
            overall=_compare(matching),
            rating_bands=bands,
            mean_best_rank=(sum(ranks) / len(ranks) if ranks else None),
            rankable_opportunities=len(ranks),
        )
    return AdjudicationReport(
        positions=tuple(
            sorted(
                positions,
                key=lambda item: (
                    item.game_id,
                    item.ply_index,
                    item.predicate.value,
                ),
            )
        ),
        predicates=reports,
    )


def action_sets(
    inputs: ScoringInputs,
) -> Mapping[tuple[int, int], Mapping[str, frozenset[int]]]:
    """Return the successful subsets the policy scorer consumes."""

    return {
        key: {
            predicate.value: match.successful_action_ids
            for predicate, match in matches.items()
        }
        for key, matches in inputs.predicates.items()
        if matches
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


def _compare(
    positions: Iterable[AdjudicatedPosition],
) -> PointReferenceComparison:
    values = tuple(positions)
    return compare_reference_rate(
        tuple(
            PairedRateObservation(
                game_id=item.game_id,
                human_success=item.human_success,
                model_success=item.model_success,
                model_probability_mass=item.policy_mass,
            )
            for item in values
        )
    )


__all__ = [
    "ADJUDICATION_VERSION",
    "AdjudicatedPosition",
    "AdjudicationReport",
    "PredicateReport",
    "action_sets",
    "build_adjudication_report",
    "merge_game_totals",
]
