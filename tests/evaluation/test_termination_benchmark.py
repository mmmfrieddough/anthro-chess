"""What the game-termination family measures, and what it refuses to measure.

Two halves. The first is the derivation: given a game that ended a particular
way, this family has to read the ending, the material behind a resignation, and
whether a draw was ever claimable, exactly. Those are the correctness gates,
and they run on constructed sequences whose endings are known rather than on
generated play that happens to reach one.

The second is the wiring: which readings reach the committed tier, which series
they land on, and — the part worth the most here — which readings report
themselves unavailable instead of writing a zero a reader would take for a
measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import chess
import pytest
import torch
from pydantic import ValidationError

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    DRAW_CLAIM_ACTION_ID,
    RESIGNATION_ACTION_ID,
    encode_move,
)
from anthro_chess.config import ConfigProvenance, ResolvedConfig
from anthro_chess.data import DecisionContext, Speed
from anthro_chess.evaluation import PoolConfig, freeze_pool
from anthro_chess.evaluation.benchmarks import benchmark_registry, run_benchmark
from anthro_chess.evaluation.cost import BENCHMARK_COST_KIND
from anthro_chess.evaluation.games import (
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameTermination,
    SeatRecord,
    build_game_record,
    termination_from_outcome,
)
from anthro_chess.evaluation.reference import ReferenceConfig
from anthro_chess.evaluation.results import (
    CheckpointReference,
    DetailStore,
    ResultsStore,
)
from anthro_chess.evaluation.results.metrics import (
    GAME_TERMINATION_FAMILY,
    TERMINATION_MIX_CONDITIONAL_DISTANCE,
    TERMINATION_MIX_POOLED_DISTANCE,
    TERMINATION_PREDICTION_PROJECTION,
    TERMINATION_PREMATURE_RESIGNATION_HUMAN_RATE,
    TERMINATION_PREMATURE_RESIGNATION_RATE,
    TERMINATION_RESIGNATION_DEFICIT_DISTANCE,
    TERMINATION_RESIGNATION_DEFICIT_MEDIAN,
    TERMINATION_RESIGNATION_MASS_AT_MOVES,
    TERMINATION_RESIGNATION_MASS_AT_RESIGNATION,
    TERMINATION_RESIGNATION_MASS_SEPARATION,
    TERMINATION_SILENT_TERMINAL_ACTIONS,
    TERMINATION_UNTIMED_NON_TERMINATION_RATE,
    MetricDirection,
    registered_metrics,
)
from anthro_chess.evaluation.termination import (
    DECLARED_MIX_NEIGHBOURS as _DECLARED,
)
from anthro_chess.evaluation.termination import (
    HUMAN_ONLY_CATEGORIES,
    MODEL_ONLY_CATEGORIES,
    TERMINATION_KIND,
    TERMINATION_MIX_CATEGORIES,
    TerminationBenchmarkConfig,
    TerminationBenchmarkError,
    TerminationBenchmarkResult,
    generated_ending,
    human_ending,
)
from anthro_chess.runtime import RuntimeConfig

CHECKPOINT = CheckpointReference(label="fixture-checkpoint", step=1)

#: Knights out and back twice, which returns the starting position for the
#: third time. A draw becomes claimable at the final position and the game has
#: not ended, which is the non-termination failure the claim action exists for.
THREEFOLD_SHUFFLE = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8")

#: White to move and stalemate available in one. The ending is a property of
#: the position rather than of anything a seat chose.
STALEMATE_FEN = "k7/8/2K5/1Q6/8/8/8/8 w - - 0 1"
STALEMATE_MOVES = ("b5b6",)

#: The last piece besides the kings is captured, so the rules end the game on
#: their own with no claim and no decision.
INSUFFICIENT_FEN = "k7/8/8/8/8/8/6n1/K6B w - - 0 1"
INSUFFICIENT_MOVES = ("h1g2",)

#: A resignation from a hopeless position: black is a queen down and to move.
LOST_FEN = "k7/8/8/8/8/8/8/K5Q1 b - - 0 40"

#: A resignation from a position black is winning, which is the guardrail's
#: whole reason to exist.
WINNING_FEN = "k5q1/8/8/8/8/8/8/K7 b - - 0 40"


@dataclass
class ShufflingRunner:
    """A stand-in policy that prefers the first legal move it is offered.

    Deterministic and terminal-action averse: it never resigns and never
    claims, which is exactly the checkpoint the silent-non-use guardrail
    exists to name.
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def predict(self, context: DecisionContext) -> torch.Tensor:
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[RESIGNATION_ACTION_ID] = -50.0
        logits[DRAW_CLAIM_ACTION_ID] = -50.0
        generator = torch.Generator().manual_seed(len(context.plies))
        return logits + torch.randn(ACTION_VOCABULARY_SIZE, generator=generator) * 0.01


@dataclass
class ResigningRunner:
    """A stand-in policy that resigns as soon as it is allowed to."""

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def predict(self, context: DecisionContext) -> torch.Tensor:
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[RESIGNATION_ACTION_ID] = 40.0
        return logits


@dataclass
class TargetAwareScorer:
    """A stand-in scorer whose resignation mass tracks the human's own action.

    It reads the batch's targets, which no real model can do. That is the
    point: it produces a reading whose direction is known in advance, so a test
    can assert that the two halves of the held-out measurement are separated
    the way they are defined to be rather than merely that both are floats.
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    #: Zero makes the scorer blind to the target, which is what the collapsed
    #: separation case needs.
    separation: float = 6.0

    def predict(self, context: DecisionContext) -> torch.Tensor:
        return torch.zeros(ACTION_VOCABULARY_SIZE)

    def action_logits(self, batch: Any) -> torch.Tensor:
        shape = (*batch.action_targets.shape, ACTION_VOCABULARY_SIZE)
        logits = torch.zeros(shape, dtype=torch.float32)
        resigned = batch.action_targets == RESIGNATION_ACTION_ID
        logits[..., RESIGNATION_ACTION_ID] = torch.where(
            resigned,
            torch.full_like(batch.action_targets, 0, dtype=torch.float32)
            + self.separation,
            torch.zeros_like(batch.action_targets, dtype=torch.float32),
        )
        return logits


def _record(
    *,
    initial_position: str,
    moves: Sequence[str],
    outcome: GameOutcome,
    rating: int = 1500,
    terminal_action_id: int | None = None,
) -> GameRecord:
    """Assemble one generated record from an explicit line and ending."""

    board = chess.Board(initial_position)
    action_ids: list[int] = []
    for move_text in moves:
        move = chess.Move.from_uci(move_text)
        assert move in board.legal_moves, f"{move_text} is illegal here"
        action_ids.append(encode_move(move))
        board.push(move)
    if terminal_action_id is not None:
        action_ids.append(terminal_action_id)

    root = chess.Board(initial_position)
    seat = SeatRecord(
        kind="model",
        label="fixture",
        seed=0,
        configuration={"target_rating": rating, "temperature": 1.0},
    )
    decisions = tuple(
        DecisionRecord(
            ply_index=index,
            slot=(
                "white" if (root.turn == chess.WHITE) == (index % 2 == 0) else "black"
            ),
            action_id=action_id,
        )
        for index, action_id in enumerate(action_ids)
    )
    return build_game_record(
        initial_position=initial_position,
        prefix_plies=0,
        action_ids=action_ids,
        white=seat,
        black=seat,
        seed=0,
        decisions=decisions,
        outcome=outcome,
    )


@pytest.fixture
def small_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the declared bandwidth so a fixture reference can support it.

    The declared value is chosen from tens of thousands of real games and a
    fixture cannot hold that many. The constant itself is asserted separately.
    """

    monkeypatch.setattr(
        "anthro_chess.evaluation.termination.DECLARED_MIX_NEIGHBOURS", 4
    )


@pytest.fixture
def pool(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> Path:
    """Freeze a pool whose test games end in a mix of ways.

    Even ply counts with a black loss leave White to move at the end, which is
    what makes the derivation attribute the resignation to the side to move and
    append the terminal action. Those are the games the held-out reading has
    any positives at all from.
    """

    rows = [
        normalized_row(
            index,
            split="test",
            plies=4 + (index % 4) * 2,
            rating=1100 + (index % 10) * 100,
            result=("0-1", "1-0", "1/2-1/2")[index % 3],
        )
        for index in range(1, 41)
    ]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    output = tmp_path / "pool"
    freeze_pool(
        ResolvedConfig(
            value=PoolConfig.model_validate(
                {
                    "pool_id": "fixture-termination",
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    return output


def _config(pool: Path, **overrides: Any) -> ResolvedConfig[TerminationBenchmarkConfig]:
    """Return a resolved suite small enough for the CPU test suite."""

    fields: dict[str, Any] = {
        "pool": str(pool),
        "runtime": RuntimeConfig(resignation_enabled=True, draw_claim_enabled=True),
        "grid": {
            "target_ratings": (1200, 1800),
            "temperatures": (1.0,),
            "seeds": (0,),
            **overrides.pop("grid", {}),
        },
        "generation": {
            "games_per_position": 1,
            "maximum_generated_plies": 6,
            "swap_colors": False,
            **overrides.pop("generation", {}),
        },
        "reference": {"resamples": 8, **overrides.pop("reference", {})},
        "held_out": {"enabled": False, **overrides.pop("held_out", {})},
    }
    fields.update(overrides)
    return ResolvedConfig(
        value=TerminationBenchmarkConfig.model_validate(fields),
        provenance=ConfigProvenance(source=None, overrides=()),
    )


def _run(
    config: ResolvedConfig[TerminationBenchmarkConfig],
    *,
    runner: Any | None = None,
    store: ResultsStore | None = None,
    detail: DetailStore | None = None,
) -> TerminationBenchmarkResult:
    return cast(
        TerminationBenchmarkResult,
        run_benchmark(
            benchmark_registry()["termination"],
            config,
            store=store,
            detail=detail,
            runner=runner or ShufflingRunner(),
            checkpoint=CHECKPOINT,
        ),
    )


# --- The shared vocabulary ------------------------------------------------


def test_both_sides_are_counted_over_one_vocabulary() -> None:
    """Every category either side can produce is in the compared vocabulary."""

    for termination in GameTermination:
        assert termination.value in TERMINATION_MIX_CATEGORIES
    for category in (*HUMAN_ONLY_CATEGORIES, *MODEL_ONLY_CATEGORIES):
        assert category in TERMINATION_MIX_CATEGORIES


def test_categories_neither_side_can_produce_stay_their_own_bucket() -> None:
    """Abandonment and the ply limit are visible rather than folded away."""

    assert "abandonment" in HUMAN_ONLY_CATEGORIES
    assert GameTermination.PLY_LIMIT.value in MODEL_ONLY_CATEGORIES
    # The two sets are genuinely disjoint: a category is either something only
    # a human platform produces or something only the harness does, never both.
    assert not set(HUMAN_ONLY_CATEGORIES) & set(MODEL_ONLY_CATEGORIES)


def test_the_rule_endings_share_their_names_across_both_vocabularies() -> None:
    """A rule ending counts as one category however it was produced."""

    from anthro_chess.data.termination import TerminationCategory

    shared = {
        GameTermination.CHECKMATE,
        GameTermination.STALEMATE,
        GameTermination.INSUFFICIENT_MATERIAL,
        GameTermination.SEVENTYFIVE_MOVES,
        GameTermination.FIVEFOLD_REPETITION,
        GameTermination.FIFTY_MOVES,
        GameTermination.THREEFOLD_REPETITION,
        GameTermination.RESIGNATION,
    }
    for termination in shared:
        assert TerminationCategory(termination.value).value == termination.value


# --- Correctness gates ----------------------------------------------------


def test_a_claimable_threefold_that_never_ends_is_reported_as_such() -> None:
    """The non-termination guardrail sees a claim the model declined to take."""

    record = _record(
        initial_position=chess.STARTING_FEN,
        moves=THREEFOLD_SHUFFLE,
        outcome=GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=True,
        ),
    )
    # The chess layer agrees the claim was genuinely available, so the reading
    # rests on exact rule logic rather than on the harness's own bookkeeping.
    assert record.board().is_repetition(3)

    ending = generated_ending(record)
    assert ending.claimable_and_unfinished is True
    assert ending.category == GameTermination.PLY_LIMIT.value


def test_a_threefold_the_model_claims_is_not_a_non_termination() -> None:
    """Claiming is the behavior the guardrail exists to reward, not punish."""

    record = _record(
        initial_position=chess.STARTING_FEN,
        moves=THREEFOLD_SHUFFLE,
        terminal_action_id=DRAW_CLAIM_ACTION_ID,
        outcome=GameOutcome(
            result="1/2-1/2",
            termination=GameTermination.THREEFOLD_REPETITION,
            adjudicated=False,
        ),
    )
    ending = generated_ending(record)
    assert ending.claimable_and_unfinished is False
    assert ending.category == GameTermination.THREEFOLD_REPETITION.value
    assert ending.selected_terminal_actions == frozenset({DRAW_CLAIM_ACTION_ID})


@pytest.mark.parametrize(
    ("initial_position", "moves", "expected"),
    [
        (STALEMATE_FEN, STALEMATE_MOVES, GameTermination.STALEMATE),
        (
            INSUFFICIENT_FEN,
            INSUFFICIENT_MOVES,
            GameTermination.INSUFFICIENT_MATERIAL,
        ),
    ],
)
def test_an_automatic_draw_is_detected_exactly(
    initial_position: str,
    moves: tuple[str, ...],
    expected: GameTermination,
) -> None:
    """A rule ending needs no claim, and the position itself proves which."""

    record = _record(
        initial_position=initial_position,
        moves=moves,
        outcome=GameOutcome(result="1/2-1/2", termination=expected, adjudicated=False),
    )
    outcome = record.board().outcome()
    assert outcome is not None
    # Result detection is exact: the chess layer's own classification of the
    # final position agrees with the ending the record carries.
    assert termination_from_outcome(outcome) is expected

    ending = generated_ending(record)
    assert ending.category == expected.value
    assert ending.claimable_and_unfinished is False
    assert ending.resignation_deficit is None


def test_a_resignation_deficit_is_read_from_the_resigning_players_side() -> None:
    """Behind is positive, which is the direction the human data was measured in."""

    record = _record(
        initial_position=LOST_FEN,
        moves=(),
        terminal_action_id=RESIGNATION_ACTION_ID,
        outcome=GameOutcome(
            result="1-0",
            termination=GameTermination.RESIGNATION,
            adjudicated=False,
        ),
    )
    ending = generated_ending(record)
    assert ending.resignation_deficit == pytest.approx(9.0)


def test_resigning_while_winning_reads_as_a_negative_deficit() -> None:
    """The premature case has to be visible in the quantity, not only in a rate."""

    record = _record(
        initial_position=WINNING_FEN,
        moves=(),
        terminal_action_id=RESIGNATION_ACTION_ID,
        outcome=GameOutcome(
            result="1-0",
            termination=GameTermination.RESIGNATION,
            adjudicated=False,
        ),
    )
    ending = generated_ending(record)
    assert ending.resignation_deficit == pytest.approx(-9.0)


def test_a_human_resignation_is_read_from_the_losing_players_side(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    """One definition of deficit, applied identically on both sides."""

    row = normalized_row(1, split="test", plies=4, result="0-1", rating=1500)
    ending, reason = human_ending(row, ReferenceConfig())
    assert reason is None
    assert ending is not None
    assert ending.category == "resignation"
    # Four plies of the shared opening leave material level, so the human side
    # reports a zero deficit rather than declining to report one.
    assert ending.resignation_deficit == pytest.approx(0.0)


def test_a_lopsided_human_game_is_excluded_rather_than_averaged(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    """A mismatch belongs to neither player's rating, so it belongs to neither."""

    row = normalized_row(1, split="test", plies=4, result="0-1", rating=1500)
    row["black_normalized_rating"] = 2400
    _, reason = human_ending(row, ReferenceConfig(maximum_rating_gap=200))
    assert reason == "rating_gap"


def test_a_reference_ending_carries_the_class_both_clock_columns_derive(
    normalized_row: Callable[..., dict[str, Any]],
) -> None:
    """A reference ending carries the class a training selection filters on.

    One minute is bullet or blitz depending on the increment, so the two rows
    differ only in that column.
    """

    def speed_of(increment_ms: int) -> Speed | None:
        row = normalized_row(
            1,
            plies=4,
            time_initial_ms=60_000,
            time_increment_ms=increment_ms,
        )
        ending, _ = human_ending(row, ReferenceConfig())
        assert ending is not None
        return ending.speed

    assert speed_of(0) is Speed.BULLET
    assert speed_of(3_000) is Speed.BLITZ


# --- Guardrails -----------------------------------------------------------


def test_a_model_that_never_resigns_reports_silent_non_use(pool: Path) -> None:
    """The opposite failure to premature resignation, named rather than implied."""

    result = _run(_config(pool))
    reading = result.reading(1.0)
    assert reading.guardrails.resignations == 0
    assert set(reading.guardrails.enabled_terminal_actions) == {
        RESIGNATION_ACTION_ID,
        DRAW_CLAIM_ACTION_ID,
    }
    assert set(reading.guardrails.silent_terminal_actions) == {
        RESIGNATION_ACTION_ID,
        DRAW_CLAIM_ACTION_ID,
    }


def test_a_model_that_never_resigns_has_no_deficit_rather_than_a_zero(
    pool: Path,
) -> None:
    """A zero deficit would read as resigning while exactly level."""

    result = _run(_config(pool))
    reading = result.reading(1.0)
    assert reading.deficit.model_median is None
    assert "resignation_deficit" in reading.unavailable
    reported = {
        found.metric for envelope in result.envelopes for found in envelope.measurements
    }
    assert TERMINATION_RESIGNATION_DEFICIT_MEDIAN.identifier not in reported
    assert TERMINATION_RESIGNATION_DEFICIT_DISTANCE.identifier not in reported


def test_a_disabled_terminal_action_is_absent_from_the_silent_count(
    pool: Path,
) -> None:
    """An action the runtime never offered is not one the model left unused."""

    result = _run(
        _config(
            pool,
            runtime=RuntimeConfig(resignation_enabled=False, draw_claim_enabled=False),
        )
    )
    reading = result.reading(1.0)
    assert reading.guardrails.enabled_terminal_actions == ()
    assert "silent_terminal_actions" in reading.unavailable
    reported = {
        found.metric for envelope in result.envelopes for found in envelope.measurements
    }
    assert TERMINATION_SILENT_TERMINAL_ACTIONS.identifier not in reported


def test_a_model_that_always_resigns_is_measured_as_premature(pool: Path) -> None:
    """Resigning from the starting position is the failure this family exists for."""

    result = _run(_config(pool), runner=ResigningRunner())
    reading = result.reading(1.0)
    guardrails = reading.guardrails
    assert guardrails.resignations == guardrails.games
    assert guardrails.premature_rate == pytest.approx(1.0)
    assert guardrails.silent_terminal_actions == (DRAW_CLAIM_ACTION_ID,)
    # The human rate is reported beside it, which is what makes a heuristic
    # material proxy readable rather than an absolute claim.
    assert guardrails.human_premature_rate is not None


def test_the_untimed_non_termination_rate_is_reported_over_every_game(
    pool: Path,
) -> None:
    """A rate needs its denominator to be the games that could have failed."""

    result = _run(_config(pool))
    guardrails = result.reading(1.0).guardrails
    assert guardrails.games > 0
    assert guardrails.untimed_non_termination_rate == pytest.approx(
        guardrails.claimable_unfinished_games / guardrails.games
    )


# --- Held-out resignation prediction --------------------------------------


def test_the_held_out_reading_separates_resignation_plies_from_move_plies(
    pool: Path,
) -> None:
    """Both halves come out of one pass, and the separation is their difference."""

    result = _run(
        _config(pool, held_out={"enabled": True}),
        runner=TargetAwareScorer(),
    )
    held_out = result.held_out
    assert held_out is not None
    assert held_out.resignation_plies > 0
    assert held_out.move_plies > held_out.resignation_plies
    assert held_out.mass_at_resignation is not None
    assert held_out.mass_at_moves is not None
    assert held_out.mass_at_resignation > held_out.mass_at_moves
    assert held_out.separation == pytest.approx(
        held_out.mass_at_resignation - held_out.mass_at_moves
    )


def test_a_policy_blind_to_the_ending_shows_no_separation(pool: Path) -> None:
    """The reading has to be able to say the model learned nothing."""

    result = _run(
        _config(pool, held_out={"enabled": True}),
        runner=TargetAwareScorer(separation=0.0),
    )
    held_out = result.held_out
    assert held_out is not None
    assert held_out.separation == pytest.approx(0.0, abs=1e-9)


def test_the_held_out_halves_carry_their_own_sample_sizes(pool: Path) -> None:
    """The two populations differ by orders of magnitude and must say so."""

    result = _run(
        _config(pool, held_out={"enabled": True}),
        runner=TargetAwareScorer(),
    )
    held_out = result.held_out
    assert held_out is not None
    envelope = _held_out_envelope(result)
    at_resignation = envelope.measurement(
        TERMINATION_RESIGNATION_MASS_AT_RESIGNATION.identifier
    )
    at_moves = envelope.measurement(TERMINATION_RESIGNATION_MASS_AT_MOVES.identifier)
    assert at_resignation is not None
    assert at_moves is not None
    assert at_resignation.sample_size == held_out.resignation_plies
    assert at_moves.sample_size == held_out.move_plies
    assert at_moves.sample_size != at_resignation.sample_size


def test_the_held_out_reading_is_scoped_by_the_content_it_scored(
    pool: Path,
) -> None:
    """Nothing was generated, so the human decisions are series identity."""

    result = _run(
        _config(pool, held_out={"enabled": True}),
        runner=TargetAwareScorer(),
    )
    envelope = _held_out_envelope(result)
    assert envelope.execution is None
    assert envelope.data is not None
    assert [digest.projection for digest in envelope.data.components] == [
        TERMINATION_PREDICTION_PROJECTION.name
    ]


def test_a_view_without_a_resignation_reports_unavailable(
    tmp_path: Path,
    normalized_row: Callable[..., dict[str, Any]],
    write_corpus: Callable[..., tuple[Path, Path]],
) -> None:
    """No positive to score against is a state, not a zero mass.

    This is also the shape a corpus whose vocabulary predates terminal actions
    takes here. Such a pool is rejected outright when it is loaded, because its
    action-vocabulary identity no longer matches; what survives that check and
    still has nothing to measure is a compatible pool holding no game that
    carries a terminal action, which is exactly this.
    """

    rows = [
        normalized_row(index, split="test", plies=6, rating=1500, result="1-0")
        for index in range(1, 13)
    ]
    normalized, manifest = write_corpus(tmp_path / "corpus", rows)
    output = tmp_path / "pool"
    freeze_pool(
        ResolvedConfig(
            value=PoolConfig.model_validate(
                {
                    "pool_id": "fixture-no-resignations",
                    "normalized": str(normalized),
                    "manifest": str(manifest),
                }
            ),
            provenance=ConfigProvenance(source=None, overrides=()),
        ),
        output,
    )
    result = _run(
        _config(output, held_out={"enabled": True}),
        runner=TargetAwareScorer(),
    )
    held_out = result.held_out
    assert held_out is not None
    assert held_out.resignation_plies == 0
    assert held_out.mass_at_resignation is None
    assert "resignation_mass_at_resignation" in held_out.unavailable
    reported = {
        found.metric for envelope in result.envelopes for found in envelope.measurements
    }
    assert TERMINATION_RESIGNATION_MASS_AT_RESIGNATION.identifier not in reported
    assert TERMINATION_RESIGNATION_MASS_SEPARATION.identifier not in reported
    # The half that does have a population is still reported, so one missing
    # reading does not take the rest of the family down with it.
    assert TERMINATION_RESIGNATION_MASS_AT_MOVES.identifier in reported


def test_a_runner_that_cannot_score_batches_says_so(pool: Path) -> None:
    """A stand-in that only plays games cannot produce the held-out half."""

    with pytest.raises(TerminationBenchmarkError, match="scores whole batches"):
        _run(_config(pool, held_out={"enabled": True}), runner=ShufflingRunner())


# --- The mix comparison ---------------------------------------------------


def test_the_mix_compares_both_sides_over_one_rating_axis(
    pool: Path,
    small_bandwidth: None,
) -> None:
    """The headline reading is the shared curve shape, not a new one."""

    result = _run(_config(pool))
    mix = result.mix("overall", 1.0)
    assert mix.human_games > 0
    assert mix.model_games > 0
    assert mix.comparison.spec.grid == (1200.0, 1800.0)
    assert 0.0 <= mix.comparison.pooled_distance <= 1.0
    categories = {share.category for share in mix.comparison.category_shares()}
    assert categories <= set(TERMINATION_MIX_CATEGORIES)


def test_the_mix_reports_how_far_its_smoother_actually_reached(
    pool: Path,
    small_bandwidth: None,
) -> None:
    """The declared neighbour count is the same whatever the reference holds.

    What decides whether the grid resolves its points is the rating span those
    neighbours occupy, so that is what the mix prints beside its distances.
    """

    from anthro_chess.interfaces.cli import _render_termination

    result = _run(_config(pool))

    comparison = result.mix("overall", 1.0).comparison
    spans = " ".join(f"±{point.bandwidth:.0f}" for point in comparison.points)
    assert f"  bandwidth   reaches {spans} rating points" in _render_termination(result)


def test_a_reference_below_the_bandwidth_is_rejected_before_a_sweep_starts(
    pool: Path,
) -> None:
    """The mix bandwidth is a neighbour count, so this cap is its radius.

    A cap below one bandwidth per grid point does not make the mix noisier; it
    makes every grid point the same neighbourhood. Rejected on the configuration
    so a suite plan catches it rather than the run that follows.
    """

    with pytest.raises(ValidationError, match="below the 2048 game"):
        _config(
            pool,
            reference={
                "view": {"name": "termination-reference", "maximum_games": 2000},
            },
        )


def test_a_reference_too_thin_for_the_mix_says_so_in_the_pool_pass(
    pool: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The declared cap cannot promise what the pool and the filter leave.

    Said while the reference is read rather than only in the unavailable lines
    the run ends with. Not an error, because the guardrails, the deficit, and
    the held-out reading need no curve at all.
    """

    result = _run(_config(pool))

    assert "below the" in caplog.text
    assert result.mixes == ()
    assert "mix:overall" in result.unavailable


def test_the_human_reference_scopes_the_mix_series(
    pool: Path,
    small_bandwidth: None,
) -> None:
    """Two references are two smoothings, so two quantities rather than two draws.

    Without this in identity a checkpoint whose endings were compared against a
    small reference would be plotted against one compared against a large one,
    and the difference between the smoothings would render as movement.
    """

    workloads = []
    for maximum_games in (16, 32):
        result = _run(
            _config(
                pool,
                reference={
                    "view": {
                        "name": "termination-reference",
                        "require_ratings": True,
                        "maximum_games": maximum_games,
                    },
                },
            )
        )
        workloads.append(result.mix("overall", 1.0).execution.workload_sha256)

    assert workloads[0] != workloads[1]


def test_the_mix_reports_each_arm_beside_its_own_qualifiers(
    pool: Path,
    small_bandwidth: None,
) -> None:
    """One arm's qualifier printed beside the other's distance is misread.

    The mix line carried a single floor — the conditional one — with the pooled
    distance immediately after it, and the null level the verdict is actually
    computed against was not printed at all. The same defect as the rollout
    table, recorded together in #172.
    """

    from anthro_chess.interfaces.cli import _render_termination

    result = _run(_config(pool))

    rendered = _render_termination(result)
    comparison = result.mix("overall", 1.0).comparison
    assert comparison.references is not None
    assert comparison.dispersions is not None
    assert (
        f"  conditional {comparison.conditional_distance:.4f}  "
        f"null {comparison.references.conditional:.4f}  "
        f"floor {comparison.dispersions.conditional_floor:.4f}"
    ) in rendered
    assert (
        f"  pooled      {comparison.pooled_distance:.4f}  "
        f"null {comparison.references.pooled:.4f}  "
        f"floor {comparison.dispersions.pooled_floor:.4f}"
    ) in rendered
    assert f"  reads as    {comparison.response.value}" in rendered


def test_a_reference_too_small_for_the_bandwidth_reports_unavailable(
    pool: Path,
) -> None:
    """A distance no reference can support is refused rather than estimated."""

    result = _run(_config(pool))
    assert result.mixes == ()
    assert "mix:overall" in result.unavailable
    assert "bandwidth" in result.unavailable["mix:overall"]


def test_two_speed_classes_land_on_different_series(
    pool: Path,
    small_bandwidth: None,
) -> None:
    """Two human populations are two questions, so they must not share a line."""

    result = _run(
        _config(
            pool,
            speeds=("overall", Speed.BLITZ),
        )
    )
    everything = result.mix("overall", 1.0)
    blitz = result.mix("blitz", 1.0)
    assert everything.execution.workload_sha256 != blitz.execution.workload_sha256
    conditional = TERMINATION_MIX_CONDITIONAL_DISTANCE.identifier
    found = [envelope.measurement(conditional) for envelope in result.envelopes]
    fingerprints = {item.fingerprint for item in found if item is not None}
    assert len(fingerprints) == 2


def test_an_unpopulated_speed_class_reports_unavailable(
    pool: Path,
    small_bandwidth: None,
) -> None:
    """A class no reference game belongs to has nothing to compare against."""

    result = _run(_config(pool, speeds=(Speed.CLASSICAL,)))
    assert result.mixes == ()
    assert "mix:classical" in result.unavailable


def test_the_declared_bandwidth_is_frozen_in_code() -> None:
    """Re-selecting per run would mean two checkpoints were measured differently."""

    assert _DECLARED == 1024


def test_a_single_rating_is_a_point_rather_than_a_curve(pool: Path) -> None:
    """A curve's axis is the rating, so one rating cannot produce one."""

    with pytest.raises(TerminationBenchmarkError, match="at least two"):
        _run(_config(pool, grid={"target_ratings": (1500,)}))


def test_a_pool_this_suite_is_not_defined_over_is_refused(pool: Path) -> None:
    """The generation a selection pins reaches the loader from here."""

    with pytest.raises(TerminationBenchmarkError, match="expected 0{64}"):
        _run(_config(pool, expected_pool_game_ids_sha256="0" * 64))


# --- Recording ------------------------------------------------------------


def test_every_reading_is_recorded_under_its_own_kind(
    tmp_path: Path,
    pool: Path,
    small_bandwidth: None,
) -> None:
    """Three units, three records, each with its own series."""

    store = ResultsStore(tmp_path / "results")
    detail = DetailStore(tmp_path / "detail")
    result = _run(
        _config(pool, held_out={"enabled": True}),
        runner=TargetAwareScorer(),
        store=store,
        detail=detail,
    )
    assert len(result.recorded_paths) == len(result.envelopes)
    assert {envelope.kind for envelope in result.envelopes} == {
        TERMINATION_KIND,
        BENCHMARK_COST_KIND,
    }
    # Every reading writes a detail payload; the cost record has none to write.
    assert len(result.detail_paths) == len(result.envelopes) - 1
    for path in result.detail_paths:
        assert path.is_file()


def test_a_greedy_reading_counts_the_endings_it_played_once(pool: Path) -> None:
    """Every ending is counted, so a replayed game would be counted again.

    Greedy seats replay one game per position, so a temperature-zero reading
    plays one replicate rather than the configured grid of them.
    """

    result = _run(
        _config(
            pool,
            grid={"temperatures": (0.0,), "seeds": (0, 1, 2)},
            generation={"games_per_position": 2},
        )
    )

    # Two ratings, one position each, neither swapped nor replayed.
    assert result.reading(0.0).games == 2


def test_the_generated_readings_are_scoped_by_the_recipe_they_were_played_under(
    pool: Path,
) -> None:
    """Whether the terminal actions were enabled decides what was measured."""

    enabled = _run(_config(pool)).reading(1.0)
    disabled = _run(
        _config(
            pool,
            runtime=RuntimeConfig(resignation_enabled=False, draw_claim_enabled=False),
        )
    ).reading(1.0)
    assert enabled.execution.workload_sha256 != disabled.execution.workload_sha256
    assert enabled.execution.workload["resignation_enabled"] is True
    assert disabled.execution.workload["resignation_enabled"] is False


def test_the_premature_threshold_is_part_of_the_declared_workload(
    pool: Path,
) -> None:
    """Moving it measures a different quantity, so it ends the series."""

    default = _run(_config(pool)).reading(1.0)
    moved = _run(
        _config(pool, guardrails={"premature_material_balance": -3.0})
    ).reading(1.0)
    assert default.execution.workload_sha256 != moved.execution.workload_sha256


def test_the_family_reports_its_defects_with_a_direction() -> None:
    """The guardrails are defects; the mix's own shape is not a target."""

    directions = {
        metric.identifier: metric.direction
        for metric in registered_metrics(GAME_TERMINATION_FAMILY.identifier)
    }
    assert (
        directions[TERMINATION_PREMATURE_RESIGNATION_RATE.identifier]
        is MetricDirection.LOWER_IS_BETTER
    )
    assert (
        directions[TERMINATION_UNTIMED_NON_TERMINATION_RATE.identifier]
        is MetricDirection.LOWER_IS_BETTER
    )
    assert (
        directions[TERMINATION_SILENT_TERMINAL_ACTIONS.identifier]
        is MetricDirection.LOWER_IS_BETTER
    )
    # The human comparison rate explains the model rate rather than improving
    # on its own, which is what keeps a heuristic proxy honest.
    assert (
        directions[TERMINATION_PREMATURE_RESIGNATION_HUMAN_RATE.identifier]
        is MetricDirection.INFORMATIONAL
    )
    assert (
        directions[TERMINATION_MIX_POOLED_DISTANCE.identifier]
        is MetricDirection.LOWER_IS_BETTER
    )


def _held_out_envelope(result: TerminationBenchmarkResult) -> Any:
    """Return the envelope carrying the held-out resignation reading."""

    for envelope in result.envelopes:
        if envelope.execution is None:
            return envelope
    raise AssertionError("no envelope was recorded for the held-out reading")
