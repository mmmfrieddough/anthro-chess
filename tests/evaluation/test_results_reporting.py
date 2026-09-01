"""The compact delta view, its absences, and its comparability verdicts."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from anthro_chess.evaluation.results import (
    DEFAULT_COVERAGE,
    AxisChange,
    BenchmarkReference,
    BridgeIndex,
    CheckpointReference,
    Comparability,
    DataComponent,
    DeltaReport,
    FamilyReport,
    MetricDelta,
    MetricDirection,
    MetricDispersion,
    MetricFamily,
    Movement,
    NoiseVerdict,
    ReportError,
    ResultEnvelope,
    SeriesGroup,
    build_bridge,
    build_delta_report,
    build_history,
    build_result,
    execution_reference,
    measurement,
    render_history,
    render_provenance,
    render_report,
    reporting,
)
from anthro_chess.evaluation.results.reporting import (
    MAXIMUM_LINE_WIDTH,
    MAXIMUM_METRIC_COLUMN_WIDTH,
    MINIMUM_METRIC_COLUMN_WIDTH,
    CheckpointSelection,
    SeedScope,
)
from anthro_chess.evaluation.seed_dispersion import (
    HealthBand,
    SeedArm,
    SeedDispersion,
)

ResultFactory = Callable[..., ResultEnvelope]
RowFactory = Callable[..., dict[str, Any]]
Digest = Callable[..., DataComponent]

BASELINE_AT = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
CURRENT_AT = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)

GENERATED_PLIES = "generated_play.mean_game_plies"


def _dispersion(
    floor: float,
    *,
    source: str | None = None,
) -> MetricDispersion:
    """Return the spread a reading stores, chosen to combine to ``floor``.

    Written from the floor back rather than forward from a measured spread,
    because these tests are about which floor binds and what the row then says.
    ``test_results_noise`` owns the arithmetic that turns two spreads into one.
    """

    bound = floor / (DEFAULT_COVERAGE * math.sqrt(2.0))
    return MetricDispersion(value=bound, bound=bound, source=source)


def _row(report: DeltaReport, metric: str) -> MetricDelta:
    for family in report.families:
        for row in family.metrics:
            if row.metric == metric:
                return row
    raise AssertionError(f"{metric} is not in the report")


def _groups(report: DeltaReport, metric: str) -> tuple[SeriesGroup, ...]:
    """Return the series a metric was reported on, in report order."""

    return tuple(
        group
        for family in report.families
        for group in family.series
        if any(row.metric == metric for row in group.metrics)
    )


def _rollout_cell(
    label: str,
    plies: float,
    *,
    arm: str = "standard-start",
    rating: int = 1500,
    temperature: float = 0.9,
    recorded_at: datetime,
) -> ResultEnvelope:
    """Build one recorded cell of a generated-play matrix.

    A rollout writes one envelope per cell, so a single run records several
    results for one checkpoint that share a metric identifier and differ only
    by declared workload.
    """

    execution = execution_reference(
        device="cpu",
        device_name="fixture",
        precision="float32",
        torch_version="2.7.0",
        platform_key="Fixture-x86",
        platform="fixture-1.2.3",
        workload={
            "positions": {"kind": arm, "count": 8},
            "target_rating": rating,
            "temperature": temperature,
            "maximum_generated_plies": 200,
        },
    )
    return build_result(
        kind="generated-play",
        benchmark=BenchmarkReference(name="generated-play", version=1),
        checkpoint=CheckpointReference(label=label, step=1),
        execution=execution,
        measurements=[
            measurement(
                GENERATED_PLIES,
                plies,
                workload=execution.workload_component(),
            )
        ],
        recorded_at=recorded_at,
    )


def _family(report: DeltaReport, identifier: str) -> FamilyReport:
    for family in report.families:
        if family.family.identifier == identifier:
            return family
    raise AssertionError(f"{identifier} is not in the report")


@pytest.fixture
def two_checkpoints(
    recorded_result: ResultFactory,
) -> tuple[ResultEnvelope, ResultEnvelope]:
    """Return two results measured over identical inputs."""

    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        mask_penalty=0.75,
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        mask_penalty=0.8,
        recorded_at=CURRENT_AT,
    )
    return baseline, current


def test_default_view_compares_the_two_most_recent_checkpoints(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    baseline, current = two_checkpoints

    report = build_delta_report([baseline, current], BridgeIndex())

    assert report.current.label == "checkpoint-b"
    assert report.baseline is not None
    assert report.baseline.label == "checkpoint-a"

    move_loss = _row(report, "held_out.move_loss")
    assert move_loss.baseline == pytest.approx(3.5)
    assert move_loss.current == pytest.approx(3.2)
    assert move_loss.delta == pytest.approx(-0.3)
    assert move_loss.movement is Movement.BETTER
    assert move_loss.comparability is Comparability.SAME_SERIES

    penalty = _row(report, "legality.mask_penalty")
    assert penalty.movement is Movement.WORSE


def test_direction_of_improvement_is_declared_rather_than_inferred(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    values = [
        measurement("legality.legal_mass", value, data=component)
        for value in (0.5, 0.9)
    ]
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[values[0]],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[values[1]],
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())

    assert _row(report, "legality.legal_mass").movement is Movement.BETTER


def test_a_family_with_no_supporting_result_is_named_with_a_reason(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    """A silently dropped family reads as "fine" when it means "not measured"."""

    report = build_delta_report(list(two_checkpoints), BridgeIndex())

    training_health = _family(report, "training-health")
    assert training_health.metrics == ()
    assert training_health.absence == "no result recorded for checkpoint-b"

    rating = _family(report, "rating-behavior")
    assert rating.absence == "no result recorded for checkpoint-b"


def test_mismatched_fingerprints_are_reported_as_incomparable(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        component=move_prediction_component([scored_row(1)]),
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        component=move_prediction_component([scored_row(1), scored_row(2)]),
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert row.comparability is Comparability.INCOMPARABLE
    assert row.delta is None
    assert row.movement is Movement.UNKNOWN
    assert row.note is not None
    assert "incomparable" in row.note
    assert "incomparable" in render_report(report)


def test_a_bridge_rejoins_a_series_and_renders_as_a_seam(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        component=move_prediction_component([scored_row(1)]),
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        component=move_prediction_component([scored_row(1), scored_row(2)]),
        recorded_at=CURRENT_AT,
    )
    bridge = build_bridge(
        from_fingerprint=baseline.measurements[0].fingerprint,
        to_fingerprint=current.measurements[0].fingerprint,
        reason="storage format change only",
        author="maintainer",
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report(
        [baseline, current],
        BridgeIndex([bridge]),
    )
    row = _row(report, "held_out.move_loss")

    assert row.comparability is Comparability.BRIDGED
    assert row.delta == pytest.approx(-0.3)
    assert row.bridges == (bridge.bridge_id,)
    assert row.note == "bridged series seam"


def test_bridging_is_symmetric_and_transitive() -> None:
    """A report's answer must not depend on which side was the baseline."""

    first = build_bridge(
        from_fingerprint="a" * 64,
        to_fingerprint="b" * 64,
        reason="format",
        author="maintainer",
        recorded_at=BASELINE_AT,
    )
    second = build_bridge(
        from_fingerprint="b" * 64,
        to_fingerprint="c" * 64,
        reason="format",
        author="maintainer",
        recorded_at=CURRENT_AT,
    )
    index = BridgeIndex([first, second])

    assert index.compare("a" * 64, "c" * 64).comparability is Comparability.BRIDGED
    assert index.compare("c" * 64, "a" * 64).comparability is Comparability.BRIDGED
    assert index.series("a" * 64) == index.series("c" * 64)
    assert index.compare("a" * 64, "d" * 64).comparability is (
        Comparability.INCOMPARABLE
    )
    assert index.compare("a" * 64, "c" * 64).bridges == (first, second)


def test_a_delta_spanning_a_training_change_is_stated_without_being_withdrawn(
    recorded_result: ResultFactory,
) -> None:
    """The caveat is the header's, and the rows keep saying which way they moved.

    The identity moves on every comparison that tests a change — an
    architecture, a learning rate, a corpus filter all reach the digest — so a
    per-row verdict keyed on it would read the same on every row of every
    report and tell a reader nothing.
    """

    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        recorded_at=BASELINE_AT,
        training_sha256="4d" * 32,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        recorded_at=CURRENT_AT,
        training_sha256="5e" * 32,
    )

    report = build_delta_report([baseline, current], BridgeIndex())

    assert report.training is AxisChange.CHANGED
    assert report.as_record()["training"] == "changed"

    row = _row(report, "held_out.move_loss")
    assert row.delta == pytest.approx(-0.3)
    assert row.movement is Movement.BETTER

    assert "Training identity: changed" in render_report(report)


def test_a_missing_training_identity_reads_as_unknown_rather_than_a_match(
    recorded_result: ResultFactory,
) -> None:
    """A reading recorded before the identity existed has not been checked.

    Most of the committed store predates the field, so an absence cannot
    confound a delta; what it must not do is pass as a verified match.
    """

    baseline = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        recorded_at=BASELINE_AT,
        training_sha256=None,
    )
    current = recorded_result(
        label="checkpoint-b",
        move_loss=3.2,
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())

    assert report.training is AxisChange.UNKNOWN
    assert _row(report, "held_out.move_loss").movement is Movement.BETTER
    assert "Training identity: not recorded on both sides" in render_report(report)


def test_a_matching_training_identity_is_stated_rather_than_left_to_assume(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    """The verifiable half of the check, and the only one a control rests on."""

    report = build_delta_report(list(two_checkpoints), BridgeIndex())

    assert report.training is AxisChange.UNCHANGED
    assert "Training identity: unchanged" in render_report(report)


def test_a_slice_cannot_change_what_the_report_claims_about_the_training(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """Whether this is a control-arm comparison is a fact about the pair.

    Read off the two checkpoints' own results rather than off the rows on
    screen, so narrowing to one metric cannot change what the report claims.
    A reading that recorded no identity does not erase one another reading of
    the same checkpoint did record, which is why both views read `unchanged`
    rather than both reading `unknown`.
    """

    component = move_prediction_component()
    identified = [
        recorded_result(
            label=label,
            measurements=[measurement("held_out.move_loss", value, data=component)],
            recorded_at=at,
        )
        for label, value, at in (
            ("checkpoint-a", 3.5, BASELINE_AT),
            ("checkpoint-b", 3.2, CURRENT_AT),
        )
    ]
    unidentified = [
        recorded_result(
            label=label,
            measurements=[measurement("legality.mask_penalty", value, data=component)],
            recorded_at=at,
            training_sha256=None,
        )
        for label, value, at in (
            ("checkpoint-a", 0.75, BASELINE_AT),
            ("checkpoint-b", 0.80, CURRENT_AT),
        )
    ]
    results = [*identified, *unidentified]

    whole = build_delta_report(results, BridgeIndex())
    sliced = build_delta_report(results, BridgeIndex(), metrics=["held_out.move_loss"])

    assert whole.training is AxisChange.UNCHANGED
    assert sliced.training is whole.training
    assert "Training identity: unchanged" in render_report(whole)
    assert "Training identity: unchanged" in render_report(sliced)


def test_a_delta_is_annotated_against_its_noise_floor(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    component = move_prediction_component()
    floor = _dispersion(0.1, source="five seeds")
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement("held_out.move_loss", 3.5, data=component, dispersion=floor)
        ],
        recorded_at=BASELINE_AT,
    )
    quiet = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement("held_out.move_loss", 3.45, data=component, dispersion=floor)
        ],
        recorded_at=CURRENT_AT,
    )
    loud = recorded_result(
        label="checkpoint-c",
        measurements=[
            measurement("held_out.move_loss", 3.0, data=component, dispersion=floor)
        ],
        recorded_at=datetime(2026, 7, 9, tzinfo=UTC),
    )

    within = build_delta_report([baseline, quiet], BridgeIndex())
    cleared = build_delta_report(
        [baseline, quiet, loud],
        BridgeIndex(),
        baseline="checkpoint-a",
    )

    assert _row(within, "held_out.move_loss").noise is NoiseVerdict.WITHIN
    assert _row(cleared, "held_out.move_loss").noise is NoiseVerdict.CLEARED


def test_a_delta_floor_is_combined_from_both_readings_rather_than_one(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """Two readings of one metric do not share a spread, so neither one floors.

    The two readings committed to this repository differ by up to two orders of
    magnitude on the same metric. Reading a delta against either alone assumes
    the other matched it; combining them is that arithmetic with the assumption
    taken out.
    """

    component = move_prediction_component()
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.5,
                data=component,
                dispersion=_dispersion(0.1, source="a thousand games"),
            )
        ],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.1,
                data=component,
                dispersion=_dispersion(0.5, source="forty games"),
            )
        ],
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    # Each reading on its own would floor the delta at its own number, and the
    # combination lands strictly between the two.
    assert row.noise_floor is not None
    assert row.noise_floor == pytest.approx(math.sqrt((0.1**2 + 0.5**2) / 2))
    assert 0.1 < row.noise_floor < 0.5
    # The wider of the two alone would have called this delta of -0.4 noise.
    assert row.noise is NoiseVerdict.CLEARED
    # Which side contributed the width is the first thing a wide floor raises.
    assert row.noise_floor_source == "a thousand games + forty games"


def test_a_cleared_delta_is_not_reported_as_a_caused_one(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """A floor read off a reading's own units cannot see training-seed noise.

    Decision 0029 measured what that costs: two arms differing only by their
    initialization seed cleared 14 of 54 floored metrics. ``cleared`` alone does
    not carry that, so the report says it.
    """

    component = move_prediction_component()
    spread = _dispersion(0.1)
    report = build_delta_report(
        [
            recorded_result(
                label="checkpoint-a",
                measurements=[
                    measurement(
                        "held_out.move_loss", 3.5, data=component, dispersion=spread
                    )
                ],
                recorded_at=BASELINE_AT,
            ),
            recorded_result(
                label="checkpoint-b",
                measurements=[
                    measurement(
                        "held_out.move_loss", 3.0, data=component, dispersion=spread
                    )
                ],
                recorded_at=CURRENT_AT,
            ),
        ],
        BridgeIndex(),
    )

    assert _row(report, "held_out.move_loss").noise is NoiseVerdict.CLEARED
    # The legend wraps to the terminal width, so it is read unwrapped here.
    legend = " ".join(render_report(report).split())
    assert "not that the change caused it" in legend
    assert "initialization seeds" in legend


def test_no_row_reads_as_an_improvement_it_cannot_separate_from_noise(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """``better`` and ``within`` on one row is a contradiction a reader acts on.

    The committed store held such a row, so this is an observed failure rather
    than a hypothetical one. The floor is what decides whether anything moved.
    """

    component = move_prediction_component()
    spread = _dispersion(1.0)
    report = build_delta_report(
        [
            recorded_result(
                label="checkpoint-a",
                measurements=[
                    measurement(
                        "held_out.move_loss", 3.5, data=component, dispersion=spread
                    )
                ],
                recorded_at=BASELINE_AT,
            ),
            recorded_result(
                label="checkpoint-b",
                measurements=[
                    measurement(
                        "held_out.move_loss", 3.2, data=component, dispersion=spread
                    )
                ],
                recorded_at=CURRENT_AT,
            ),
        ],
        BridgeIndex(),
    )
    row = _row(report, "held_out.move_loss")

    assert row.noise is NoiseVerdict.WITHIN
    assert row.movement is Movement.UNCHANGED
    # The delta itself survives, so a consistent small regression stays visible.
    assert row.delta == pytest.approx(-0.3)


def test_a_floor_only_one_side_offers_does_not_qualify_the_delta(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """A floor beside one operand says nothing about the difference.

    The rating ladder is where this arrives. A seat that scored nothing or
    scored everything has no finite maximum-likelihood rating, so its number is
    a declared bound and every resample of it reproduces the bound; the reading
    withholds a floor for it deliberately. Judging a delta against the other
    checkpoint's floor would call a difference from a number that was never an
    estimate a finding.
    """

    component = move_prediction_component()
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[measurement("held_out.move_loss", 3.5, data=component)],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.0,
                data=component,
                dispersion=_dispersion(
                    0.1,
                    source="eight re-measurements of one checkpoint",
                ),
            )
        ],
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    # A delta of -0.5 against a floor of 0.1 would have read as cleared.
    assert row.noise is NoiseVerdict.UNKNOWN
    assert row.noise_floor is None
    assert row.noise_floor_source is None
    assert "unknown" in render_report(report)


def test_a_metric_that_can_carry_no_sampling_floor_says_so_rather_than_unknown(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    """ "Nobody characterized one" and "one cannot exist" are different states.

    The cross-conditioning match rate counts rating slices, so no resampling of
    games estimates its dispersion. Reporting that as ``unknown`` beside a
    metric merely awaiting a characterization would set a reader to work that
    cannot be done.
    """

    component = move_prediction_component()
    metric = "dependency.rating_cross_conditioning_match_rate"
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement(metric, 0.25, data=component, dispersion=_dispersion(0.1))
        ],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement(metric, 0.75, data=component, dispersion=_dispersion(0.1))
        ],
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())
    row = _row(report, metric)

    # Both readings carry a sampling dispersion and the pair is still refused,
    # because the declaration is about what resampling can estimate rather than
    # about whether anybody produced a number.
    assert row.noise is NoiseVerdict.UNQUALIFIABLE
    assert row.noise_floor is None
    assert row.noise_floor_source is None
    rendered = render_report(report)
    assert "unqualifiable" in rendered
    # The legend wraps to the terminal width, so it is read unwrapped here.
    assert "no sampling floor can exist" in " ".join(rendered.split())


def test_an_unknown_noise_floor_is_stated_rather_than_assumed(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    report = build_delta_report(list(two_checkpoints), BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert row.noise is NoiseVerdict.UNKNOWN
    # An uncharacterized floor is not a floor of zero, which would license
    # every delta as a finding.
    assert row.noise_floor is None
    assert row.noise_floor_source is None
    assert "unknown" in render_report(report)


def test_a_delta_inside_its_floor_is_shown_rather_than_hidden(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
) -> None:
    # A consistent small regression only stays visible if a within-floor delta
    # is still reported with its value. The change column reads ``unchanged``
    # because the floor is what decides whether anything moved, and the reader
    # holding both operands can still see which way the number went.
    component = move_prediction_component()
    baseline = recorded_result(
        label="checkpoint-a",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.5,
                data=component,
                dispersion=_dispersion(10.0),
            )
        ],
        recorded_at=BASELINE_AT,
    )
    current = recorded_result(
        label="checkpoint-b",
        measurements=[
            measurement(
                "held_out.move_loss",
                3.2,
                data=component,
                dispersion=_dispersion(10.0),
            )
        ],
        recorded_at=CURRENT_AT,
    )

    report = build_delta_report([baseline, current], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert row.noise is NoiseVerdict.WITHIN
    assert row.delta == pytest.approx(-0.3)
    assert row.movement is Movement.UNCHANGED
    assert "-0.3" in render_report(report)


def test_the_first_recorded_checkpoint_reports_without_a_baseline(
    recorded_result: ResultFactory,
) -> None:
    report = build_delta_report([recorded_result()], BridgeIndex())
    row = _row(report, "held_out.move_loss")

    assert report.baseline is None
    assert row.baseline is None
    assert row.delta is None
    assert row.note == "no baseline recorded"
    assert "Baseline: none" in render_report(report)


def test_a_slice_reports_only_what_it_was_asked_about(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    report = build_delta_report(
        list(two_checkpoints),
        BridgeIndex(),
        families=["legality"],
    )

    assert [family.family.identifier for family in report.families] == ["legality"]

    by_metric = build_delta_report(
        list(two_checkpoints),
        BridgeIndex(),
        metrics=["held_out.move_loss"],
    )
    assert [row.metric for row in by_metric.families[0].metrics] == [
        "held_out.move_loss"
    ]


def test_an_unknown_selection_is_refused(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    with pytest.raises(ReportError, match="no result is recorded"):
        build_delta_report(
            list(two_checkpoints),
            BridgeIndex(),
            current="checkpoint-z",
        )
    with pytest.raises(ReportError, match="unknown metric"):
        build_delta_report(
            list(two_checkpoints),
            BridgeIndex(),
            metrics=["held_out.nonexistent"],
        )
    with pytest.raises(ReportError, match="no recorded results"):
        build_delta_report([], BridgeIndex())


def test_history_marks_series_breaks_and_bridged_seams(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    small = move_prediction_component([scored_row(1)])
    grown = move_prediction_component([scored_row(1), scored_row(2)])
    first = recorded_result(
        label="checkpoint-a",
        move_loss=3.5,
        component=small,
        recorded_at=BASELINE_AT,
    )
    second = recorded_result(
        label="checkpoint-b",
        move_loss=3.4,
        component=small,
        recorded_at=datetime(2026, 7, 4, tzinfo=UTC),
    )
    third = recorded_result(
        label="checkpoint-c",
        move_loss=3.3,
        component=grown,
        recorded_at=CURRENT_AT,
    )

    broken = build_history([first, second, third], BridgeIndex(), "held_out.move_loss")
    assert [point.starts_new_series for point in broken.points] == [
        False,
        False,
        True,
    ]

    bridge = build_bridge(
        from_fingerprint=second.measurements[0].fingerprint,
        to_fingerprint=third.measurements[0].fingerprint,
        reason="storage format change only",
        author="maintainer",
        recorded_at=CURRENT_AT,
    )
    bridged = build_history(
        [first, second, third],
        BridgeIndex([bridge]),
        "held_out.move_loss",
    )
    assert [point.bridged_from_previous for point in bridged.points] == [
        False,
        False,
        True,
    ]
    rendered = render_history(bridged)
    assert "~ " in rendered
    assert "bridged seam" in rendered


def test_machine_readable_output_is_deterministic(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    first = build_delta_report(list(two_checkpoints), BridgeIndex()).as_record()
    second = build_delta_report(
        list(reversed(two_checkpoints)),
        BridgeIndex(),
    ).as_record()

    assert first == second
    assert first["current"] == {
        "label": "checkpoint-b",
        "recorded_at": CURRENT_AT.isoformat(),
        "results": 1,
    }


def test_provenance_differences_are_available_behind_an_option(
    recorded_result: ResultFactory,
) -> None:
    baseline = recorded_result(label="checkpoint-a", recorded_at=BASELINE_AT)
    current = recorded_result(
        label="checkpoint-b",
        kind="held-out-prediction",
        recorded_at=CURRENT_AT,
    )
    changed = current.model_copy(
        update={
            "environment": current.environment.model_copy(
                update={"package_version": "9.9.9"}
            )
        }
    )

    report = build_delta_report([baseline, changed], BridgeIndex())
    rendered = render_provenance(report)

    assert [difference.field for difference in report.provenance] == ["package_version"]
    assert "9.9.9" in rendered


def test_every_cell_of_a_matrix_is_reported_rather_than_one_of_them() -> None:
    """One envelope per cell must not collapse into one arbitrary row.

    A suite shares one ``recorded_at``, so choosing the most recent measurement
    of a metric broke the tie on ``result_id`` and rendered one cell as though
    it were the checkpoint's value.
    """

    report = build_delta_report(
        [
            _rollout_cell(
                "checkpoint-a", 60.0, arm="standard-start", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-a", 90.0, arm="human-prefix", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-b", 64.0, arm="standard-start", recorded_at=CURRENT_AT
            ),
            _rollout_cell(
                "checkpoint-b", 88.0, arm="human-prefix", recorded_at=CURRENT_AT
            ),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    groups = _groups(report, GENERATED_PLIES)
    assert [group.label for group in groups] == [
        "positions.kind=human-prefix",
        "positions.kind=standard-start",
    ]
    values = {
        group.label: (group.metrics[0].baseline, group.metrics[0].current)
        for group in groups
    }
    assert values["positions.kind=human-prefix"] == (90.0, 88.0)
    assert values["positions.kind=standard-start"] == (60.0, 64.0)
    # Each cell is its own series, so the rows have to be distinguishable in
    # the record without reconstructing the grouping.
    assert len({group.metrics[0].series for group in groups}) == 2


def test_a_series_label_names_only_what_tells_the_cells_apart() -> None:
    """A cell's whole declared workload above every row carries no information."""

    report = build_delta_report(
        [
            _rollout_cell("checkpoint-a", 60.0, rating=1200, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-a", 70.0, rating=1800, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-b", 62.0, rating=1200, recorded_at=CURRENT_AT),
            _rollout_cell("checkpoint-b", 74.0, rating=1800, recorded_at=CURRENT_AT),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    labels = [group.label for group in _groups(report, GENERATED_PLIES)]

    assert labels == ["target_rating=1200", "target_rating=1800"]
    # The fields the two cells share stay in the envelope rather than in a
    # label that has to fit above a table.
    assert not any("temperature" in (label or "") for label in labels)


def test_a_cell_measured_for_only_one_checkpoint_is_named_rather_than_dropped() -> None:
    report = build_delta_report(
        [
            _rollout_cell("checkpoint-a", 60.0, rating=1200, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-b", 62.0, rating=1200, recorded_at=CURRENT_AT),
            _rollout_cell("checkpoint-b", 74.0, rating=1800, recorded_at=CURRENT_AT),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    added = next(
        group
        for group in _groups(report, GENERATED_PLIES)
        if group.label == "target_rating=1800"
    )
    row = added.metrics[0]

    assert row.current == 74.0
    assert row.baseline is None
    assert row.delta is None
    assert row.note == "not measured for checkpoint-a"


def test_one_workload_replacing_another_is_reported_as_a_workload_change() -> None:
    """The sequential case keeps its diagnosis rather than becoming two halves.

    Splitting a family whose workload simply moved would report two half-rows
    that never name the cause, so exactly one unmatched workload on each side
    is paired instead. Anything looser would be guessing which cell of a matrix
    succeeded which.
    """

    report = build_delta_report(
        [
            _rollout_cell("checkpoint-a", 60.0, rating=1200, recorded_at=BASELINE_AT),
            _rollout_cell("checkpoint-b", 74.0, rating=1800, recorded_at=CURRENT_AT),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    (group,) = _groups(report, GENERATED_PLIES)
    row = group.metrics[0]

    assert group.label is None
    assert row.comparability is Comparability.INCOMPARABLE
    assert row.delta is None
    assert row.note is not None
    assert "the declared workload changed" in row.note


def test_a_metric_on_several_series_says_so_rather_than_showing_one(
    recorded_result: ResultFactory,
    scored_row: RowFactory,
    move_prediction_component: Digest,
) -> None:
    """A metric that declares no workload can still be hiding a second series.

    Nothing can be compared across them, so the most recent is the right one to
    show; what would be wrong is showing it as though it were the only one.
    """

    small = move_prediction_component([scored_row(1)])
    grown = move_prediction_component([scored_row(1), scored_row(2)])
    report = build_delta_report(
        [
            recorded_result(
                label="checkpoint-a",
                move_loss=3.5,
                component=grown,
                recorded_at=BASELINE_AT,
            ),
            recorded_result(
                label="checkpoint-b",
                move_loss=9.9,
                component=small,
                recorded_at=CURRENT_AT,
            ),
            recorded_result(
                label="checkpoint-b",
                move_loss=3.2,
                component=grown,
                recorded_at=datetime(2026, 7, 9, 12, 0, tzinfo=UTC),
            ),
        ],
        BridgeIndex(),
        metrics=["held_out.move_loss"],
    )
    row = _row(report, "held_out.move_loss")

    assert row.current == pytest.approx(3.2)
    assert row.note == "2 series recorded for this checkpoint; showing the most recent"


def test_the_text_view_states_which_series_each_row_belongs_to() -> None:
    report = build_delta_report(
        [
            _rollout_cell(
                "checkpoint-a", 60.0, arm="standard-start", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-a", 90.0, arm="human-prefix", recorded_at=BASELINE_AT
            ),
            _rollout_cell(
                "checkpoint-b", 64.0, arm="standard-start", recorded_at=CURRENT_AT
            ),
            _rollout_cell(
                "checkpoint-b", 88.0, arm="human-prefix", recorded_at=CURRENT_AT
            ),
        ],
        BridgeIndex(),
        metrics=[GENERATED_PLIES],
    )

    rendered = render_report(report)

    assert "[series: positions.kind=standard-start]" in rendered
    assert "[series: positions.kind=human-prefix]" in rendered
    assert rendered.count(GENERATED_PLIES) == 2
    assert max(len(line) for line in rendered.splitlines()) <= 120


def test_a_single_series_family_renders_without_a_series_header(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    """The benchmark writing one result per checkpoint gains no new lines."""

    report = build_delta_report(list(two_checkpoints), BridgeIndex())

    assert all(
        group.label is None for family in report.families for group in family.series
    )
    assert "[series:" not in render_report(report)


def test_the_default_text_view_stays_readable(
    two_checkpoints: tuple[ResultEnvelope, ResultEnvelope],
) -> None:
    rendered = render_report(build_delta_report(list(two_checkpoints), BridgeIndex()))

    assert "held_out.move_loss" in rendered
    assert "legality" in rendered
    # A family still awaiting its first metric is named rather than dropped,
    # but collapsed onto one wrapped line so registering families ahead of
    # their benchmarks cannot grow the default view.
    assert "awaiting a first metric:" in rendered
    assert "rating-behavior" in rendered
    # An absence that is about this checkpoint rather than about the plan keeps
    # its own line, because it is the actionable one.
    assert "absent: no result recorded for" in rendered
    assert max(len(line) for line in rendered.splitlines()) <= 120
    # A ratchet rather than a round number: it is the current height, so a
    # change that grows the default view has to be a deliberate one. Two lines
    # each belong to benchmark cost, decision decomposition, the puzzle-backed
    # rating family, generated play, game termination, novelty, and training
    # efficiency, which have metrics but no result in this fixture; a family
    # gets its own actionable absence section as soon as it has a metric to be
    # absent. Two further lines state the training identity and what the
    # baseline's recorded seed spread does here, both headers rather than
    # per-row annotations for exactly this reason.
    assert len(rendered.splitlines()) <= 37


def _width_report(*identifiers: str) -> DeltaReport:
    """Build a report whose rows carry exactly the given identifiers.

    Assembled directly rather than measured, because the property under test
    belongs to the renderer: what the column has to do is hold whatever
    identifiers it is handed, including ones no benchmark has registered yet.
    """

    rows = tuple(
        MetricDelta(
            metric=identifier,
            family="legality",
            direction=MetricDirection.LOWER_IS_BETTER,
            baseline=0.5,
            current=0.4,
            delta=-0.1,
            comparability=Comparability.SAME_SERIES,
            movement=Movement.BETTER,
            noise=NoiseVerdict.NOT_APPLICABLE,
            noise_floor=None,
            noise_floor_source=None,
            seed_floor=None,
            seed=NoiseVerdict.NOT_APPLICABLE,
            bridges=(),
            note=None,
        )
        for identifier in identifiers
    )
    return DeltaReport(
        baseline=CheckpointSelection(
            label="checkpoint-a", recorded_at=BASELINE_AT, results=1
        ),
        current=CheckpointSelection(
            label="checkpoint-b", recorded_at=CURRENT_AT, results=1
        ),
        families=(
            FamilyReport(
                family=MetricFamily(
                    identifier="legality",
                    title="Legality",
                    summary="fixture family",
                ),
                series=(SeriesGroup(workload=None, label=None, metrics=rows),),
                absence=None,
            ),
        ),
        provenance=(),
    )


def _header(rendered: str) -> str:
    return next(
        line for line in rendered.splitlines() if line.lstrip().startswith("metric ")
    )


def test_a_long_identifier_does_not_push_its_row_out_of_alignment() -> None:
    """The column widens to the longest name present, header included.

    Every column right of the identifier shifts on an overflowing row, so the
    default view stops scanning cleanly exactly where a reader is comparing
    numbers down a column.
    """

    rendered = render_report(
        _width_report(
            "legality.mask_penalty",
            "x" * MAXIMUM_METRIC_COLUMN_WIDTH,
        )
    )

    # The direction column is the first thing right of the identifier, so it is
    # where a shifted row shows up first.
    directions = {
        line.index("lower") for line in rendered.splitlines() if "lower" in line
    }

    assert directions == {_header(rendered).index("better")}


def test_the_widest_permitted_identifier_still_fits_the_line() -> None:
    """The registry's budget is exactly what keeps the header on one line."""

    rendered = render_report(_width_report("x" * MAXIMUM_METRIC_COLUMN_WIDTH))

    assert len(_header(rendered)) == MAXIMUM_LINE_WIDTH


def test_short_identifiers_keep_the_column_at_its_minimum() -> None:
    """A report of short names keeps the table's usual shape.

    Sizing from what is present is about making room, not about reflowing the
    common report to whichever family happens to be selected.
    """

    rendered = render_report(_width_report("legality.legal_mass"))

    indent, separator = len("  "), len(" ")
    assert _header(rendered).index("better") == (
        indent + MINIMUM_METRIC_COLUMN_WIDTH + separator
    )


def _unwrapped(report: DeltaReport) -> str:
    """Return the rendered report with its line wrapping undone.

    The headers wrap at the report's own width, so a phrase they state can be
    split across two lines without anything about it having changed.
    """

    return " ".join(render_report(report).split())


def _seed_dispersion(
    training_scope: str,
    *,
    floor: float,
    horizon_steps: int = 8000,
    health_center: float = 0.5,
    health_spread: float = 0.01,
    workload: str = "",
) -> SeedDispersion:
    """Return a characterization whose ``held_out.move_loss`` floor is ``floor``.

    Written from the floor back, the way ``_dispersion`` above is and for the
    same reason: what these tests are about is which floor binds and what the
    header then says. ``test_seed_dispersion`` owns the arithmetic.
    """

    spread = floor / (DEFAULT_COVERAGE * math.sqrt(2.0))
    return SeedDispersion(
        training_sha256=training_scope,
        horizon_steps=horizon_steps,
        arms=tuple(
            SeedArm(
                run_id=f"arm-{seed}",
                seed=seed,
                checkpoint=f"arm-{seed}-step-{horizon_steps:08d}",
                training_seconds=3600.0,
            )
            for seed in (17, 29, 43)
        ),
        metrics={
            "held_out.move_loss": {
                workload: MetricDispersion(value=spread, bound=spread)
            }
        },
        health={
            "training_health.gradient_norm": HealthBand(
                center=health_center,
                dispersion=MetricDispersion(
                    value=health_spread,
                    bound=health_spread,
                ),
            )
        },
        scoring_seconds=600.0,
        measured_at=BASELINE_AT,
    )


def _seed_pair(
    recorded_result: ResultFactory,
    component: DataComponent,
    *,
    current_loss: float,
    health: float | None = None,
    current_step: int = 8000,
) -> list[ResultEnvelope]:
    """Return a baseline and a current reading, optionally with a health one."""

    reading = _dispersion(1e-9)
    results = [
        recorded_result(
            label="checkpoint-a",
            measurements=[
                measurement(
                    "held_out.move_loss", 3.5, data=component, dispersion=reading
                )
            ],
            recorded_at=BASELINE_AT,
        ),
        recorded_result(
            label="checkpoint-b",
            step=current_step,
            measurements=[
                measurement(
                    "held_out.move_loss",
                    current_loss,
                    data=component,
                    dispersion=reading,
                )
            ],
            recorded_at=CURRENT_AT,
        ),
    ]
    if health is not None:
        results.append(
            recorded_result(
                label="checkpoint-b",
                step=current_step,
                kind="training-health",
                # A health reading has no data dependency, so it carries no
                # component: it is about the optimizer step rather than about
                # anything scored.
                measurements=[measurement("training_health.gradient_norm", health)],
                recorded_at=CURRENT_AT,
            )
        )
    return results


def test_a_delta_the_control_arms_produce_among_themselves_reads_as_unchanged(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """The benchmark floor cannot see the training run; this one can.

    A delta larger than benchmark noise is a real difference between two
    models. Whether the change under test produced it is a different question,
    and until a configuration's seed spread is recorded nothing here can answer
    it. Where one is, a delta inside it is not an improvement.
    """

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(training_scope, floor=0.2),
    )

    report = build_delta_report(
        _seed_pair(recorded_result, component, current_loss=3.4, health=0.5),
        BridgeIndex(),
    )

    row = _row(report, "held_out.move_loss")
    assert row.noise is NoiseVerdict.CLEARED
    assert row.seed is NoiseVerdict.WITHIN
    assert row.movement is Movement.UNCHANGED
    assert row.seed_floor == pytest.approx(0.2)
    assert report.seed_floor.scope is SeedScope.APPLIED
    assert "Seed floor: applied" in _unwrapped(report)
    assert "seed within" in render_report(report)


def test_a_delta_outside_both_floors_clears_both(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """Clearing both is what distinguishes a change from a different run."""

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(training_scope, floor=0.05),
    )

    report = build_delta_report(
        _seed_pair(recorded_result, component, current_loss=3.0, health=0.5),
        BridgeIndex(),
    )

    row = _row(report, "held_out.move_loss")
    assert row.noise is NoiseVerdict.CLEARED
    assert row.seed is NoiseVerdict.CLEARED
    assert row.movement is Movement.BETTER


def test_a_configuration_with_no_recorded_spread_says_so(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    training_scope: str,
) -> None:
    """A floor that could be found approximately would apply to anything.

    The default lookup is left in place here rather than stubbed: no
    characterization is checked in for the fixture identity, which is the
    negative case this asserts.
    """

    component = move_prediction_component()
    report = build_delta_report(
        _seed_pair(recorded_result, component, current_loss=3.4),
        BridgeIndex(),
    )

    row = _row(report, "held_out.move_loss")
    assert report.seed_floor.scope is SeedScope.NO_RECORD
    assert report.seed_floor.training_sha256 == training_scope
    assert row.seed_floor is None
    assert row.movement is Movement.BETTER
    assert "Seed floor: none recorded" in _unwrapped(report)
    # No row carries a seed column, so nothing below the header mentions one.
    assert " seed " not in render_report(report)


def test_a_reading_taken_at_another_horizon_is_not_floored_by_this_one(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """The horizon sits outside the identity, so the key alone does not scope it.

    A cooldown branched at a different horizon carries the trunk's training
    identity by construction, which is what lets a branch match its trunk. It
    has not been shown to share the trunk's spread.
    """

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(
            training_scope, floor=0.2, horizon_steps=8000
        ),
    )

    report = build_delta_report(
        _seed_pair(
            recorded_result,
            component,
            current_loss=3.4,
            health=0.5,
            current_step=4000,
        ),
        BridgeIndex(),
    )

    assert report.seed_floor.scope is SeedScope.HORIZON
    assert report.seed_floor.off_horizon_steps == (4000,)
    assert _row(report, "held_out.move_loss").seed_floor is None
    assert "readings were taken at step(s) 4000" in _unwrapped(report)


def test_an_arm_whose_training_health_departs_is_outside_the_floors_scope(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """Instability widens an arm's spread past what the floor was measured on.

    The failure is one-directional, which is what makes it worth checking: an
    unstable arm's readings scatter further than stable ones, so the floor reads
    too narrow and noise clears it. Quoting the floor here would be at its most
    confident exactly where it is least applicable.
    """

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(
            training_scope, floor=0.2, health_center=0.5
        ),
    )

    report = build_delta_report(
        _seed_pair(recorded_result, component, current_loss=3.4, health=9.0),
        BridgeIndex(),
    )

    assert report.seed_floor.scope is SeedScope.DEPARTED
    assert report.seed_floor.departures == ("training_health.gradient_norm",)
    assert _row(report, "held_out.move_loss").seed_floor is None
    assert _row(report, "held_out.move_loss").movement is Movement.BETTER
    assert "sits outside what those arms showed" in _unwrapped(report)


def test_an_arm_that_recorded_no_health_is_floored_but_said_to_be_unverified(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """Not having been shown to depart is not the same as having been shown not to.

    Reported the way an unrecorded training identity is, rather than by
    withholding the floor: an arm nobody measured the health of is the ordinary
    case for a reading taken before the cadence existed.
    """

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(training_scope, floor=0.2),
    )

    report = build_delta_report(
        _seed_pair(recorded_result, component, current_loss=3.4),
        BridgeIndex(),
    )

    assert report.seed_floor.scope is SeedScope.UNVERIFIED
    assert _row(report, "held_out.move_loss").seed is NoiseVerdict.WITHIN
    assert "unverified rather than established" in _unwrapped(report)


def test_a_reading_that_recorded_no_step_is_not_taken_to_be_on_the_horizon(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """Silence about where a reading was taken is not agreement that it matches.

    Most of the committed store predates the fields a reading now carries, so a
    result with a training identity and no step is the ordinary old record
    rather than a contrived one.
    """

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(training_scope, floor=0.2),
    )
    results = _seed_pair(recorded_result, component, current_loss=3.4, health=0.5)
    results[0] = recorded_result(
        label="checkpoint-a",
        step=None,
        measurements=[
            measurement(
                "held_out.move_loss", 3.5, data=component, dispersion=_dispersion(1e-9)
            )
        ],
        recorded_at=BASELINE_AT,
    )

    report = build_delta_report(results, BridgeIndex())

    assert report.seed_floor.scope is SeedScope.HORIZON
    assert report.seed_floor.unrecorded_steps == 1
    assert _row(report, "held_out.move_loss").seed_floor is None
    assert "1 reading(s) recorded no step" in _unwrapped(report)


def test_health_the_characterization_has_no_band_for_verifies_nothing(
    recorded_result: ResultFactory,
    move_prediction_component: Digest,
    monkeypatch: pytest.MonkeyPatch,
    training_scope: str,
) -> None:
    """A comparison of no readings is unverified rather than passed.

    A characterization taken before a health metric was recorded has no band for
    it, so an arm reporting only that metric has had nothing checked. Reading
    the empty departure list as agreement would say the arm was shown to be in
    scope when nothing was compared.
    """

    component = move_prediction_component()
    monkeypatch.setattr(
        reporting,
        "seed_dispersion_for",
        lambda digest, **_: _seed_dispersion(training_scope, floor=0.2),
    )
    results = _seed_pair(recorded_result, component, current_loss=3.4)
    results.append(
        recorded_result(
            label="checkpoint-b",
            kind="training-health",
            measurements=[measurement("training_health.clip_rate", 0.9)],
            recorded_at=CURRENT_AT,
        )
    )

    report = build_delta_report(results, BridgeIndex())

    assert report.seed_floor.scope is SeedScope.UNVERIFIED
    assert "unverified rather than established" in _unwrapped(report)
