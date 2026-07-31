from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

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
from anthro_chess.config import load_config
from anthro_chess.data import DecisionContext, build_decision_context
from anthro_chess.data import encoding as encoding_module
from anthro_chess.runtime import (
    ActionSelectionError,
    DrawClaimAction,
    GameSession,
    MoveAction,
    ResignationAction,
    RuntimeConfig,
    SessionStateError,
)
from anthro_chess.runtime import session as session_module


@dataclass
class StubRunner:
    logits: torch.Tensor
    contexts: list[DecisionContext] = field(default_factory=list)

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.contexts.append(context)
        return self.logits.clone()


def test_white_session_builds_full_context_and_applies_only_a_legal_move() -> None:
    runner = StubRunner(_ranked_logits("e2e4", illegal="e2e5"))
    session = GameSession(
        runner,
        config=RuntimeConfig(target_rating=1725, temperature=0.0),
    )

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.action_id == encode_move(chess.Move.from_uci("e2e4"))
    assert action.move == chess.Move.from_uci("e2e4")
    assert session.move_history == (action.move,)
    assert runner.contexts[0].target_rating == 1725
    assert len(runner.contexts[0].plies) == 1
    assert all(ply.time_initial_ms is None for ply in runner.contexts[0].plies)


def test_black_session_observes_both_players_without_an_opponent_rating() -> None:
    runner = StubRunner(_ranked_logits("c7c5"))
    session = GameSession(
        runner,
        config=RuntimeConfig(target_rating=1400, temperature=0.0),
    )
    white_move = chess.Move.from_uci("e2e4")
    session.apply_move(white_move)

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("c7c5")
    assert session.move_history == (white_move, action.move)
    context = runner.contexts[0]
    assert context.target_rating == 1400
    assert len(context.plies) == 2
    assert all(not hasattr(ply, "target_rating") for ply in context.plies)


def test_greedy_mask_excludes_illegal_moves_and_disabled_resignation() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e5"))] = 100.0
    logits[RESIGNATION_ACTION_ID] = 90.0
    logits[encode_move(chess.Move.from_uci("g1f3"))] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0),
    )

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("g1f3")
    assert action.move in chess.Board().legal_moves


def test_enabled_resignation_is_preserved_and_ends_the_session() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[RESIGNATION_ACTION_ID] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0, resignation_enabled=True),
    )

    action = session.choose_action()

    assert action == ResignationAction()
    assert session.resigned_by == chess.WHITE
    assert session.is_terminal
    assert session.move_history == ()
    with pytest.raises(SessionStateError, match="terminal"):
        session.choose_action()


def test_an_enabled_draw_claim_needs_the_rules_to_allow_it() -> None:
    """The dial cannot make an unclaimable position claimable."""

    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[DRAW_CLAIM_ACTION_ID] = 100.0
    logits[encode_move(chess.Move.from_uci("g1f3"))] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0, draw_claim_enabled=True),
    )

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("g1f3")
    assert session.claimed_draw_by is None


def test_an_available_draw_claim_is_selectable_and_ends_the_session() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[DRAW_CLAIM_ACTION_ID] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0, draw_claim_enabled=True),
        moves=_repeated_position_moves(),
    )

    action = session.choose_action()

    assert action == DrawClaimAction()
    assert session.claimed_draw_by == chess.WHITE
    assert session.is_terminal
    # A claim moves nothing, so the game keeps exactly the plies it had.
    assert len(session.move_history) == len(_repeated_position_moves())
    with pytest.raises(SessionStateError, match="terminal"):
        session.choose_action()


def test_a_claimable_position_stays_playable_until_somebody_claims_it() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("d2d4"))] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0, draw_claim_enabled=True),
        moves=_repeated_position_moves(),
    )

    assert not session.is_terminal

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert not session.is_terminal


def test_a_new_game_clears_a_claimed_draw() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[DRAW_CLAIM_ACTION_ID] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0, draw_claim_enabled=True),
        moves=_repeated_position_moves(),
    )
    session.choose_action()

    session.reset()

    assert session.claimed_draw_by is None
    assert not session.is_terminal


def _repeated_position_moves() -> tuple[chess.Move, ...]:
    """Return moves reaching the starting position for the third time."""

    return tuple(
        chess.Move.from_uci(text)
        for text in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8")
    )


def test_seeded_sampling_is_repeatable_and_reset_restarts_the_stream() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    first = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=91),
    )
    second = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=91),
    )

    first_action = first.choose_action()
    second_action = second.choose_action()
    first.reset()
    repeated_action = first.choose_action()

    assert first_action == second_action == repeated_action


def test_position_sync_preserves_the_active_random_stream() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    baseline = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=31),
    )
    synced = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=31),
    )

    baseline_first = _chosen_move(baseline)
    reply = next(iter(baseline.board.legal_moves))
    baseline.apply_move(reply)
    baseline_second = _chosen_move(baseline)

    synced_first = _chosen_move(synced)
    # Advancing the position through synchronization must not reseed or rewind
    # the stream, so the continued draw matches the uninterrupted baseline.
    outcome = synced.sync_position(moves=(synced_first, reply))
    synced_second = _chosen_move(synced)

    assert synced_first == baseline_first
    assert outcome.reused_prefix_plies == 1
    assert outcome.replaced is False
    assert synced_second == baseline_second


def test_temperature_zero_is_greedy_for_every_seed_mode() -> None:
    logits = _ranked_logits("g1f3", illegal="e2e5")
    for seed in (None, 0, 4242):
        session = GameSession(
            StubRunner(logits),
            config=RuntimeConfig(temperature=0.0, seed=seed),
        )
        assert _chosen_move(session) == chess.Move.from_uci("g1f3")


def test_fixed_seed_reproduces_while_distinct_seeds_diverge() -> None:
    logits = torch.arange(ACTION_VOCABULARY_SIZE, dtype=torch.float32)
    logits = logits / logits.max()

    def first_move(seed: int) -> chess.Move:
        session = GameSession(
            StubRunner(logits),
            config=RuntimeConfig(temperature=1.0, seed=seed),
        )
        return _chosen_move(session)

    assert first_move(11) == first_move(11)
    assert len({first_move(seed).uci() for seed in range(12)}) > 1


def test_fresh_default_draws_a_new_stream_each_new_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn = iter([101, 202, 303])
    monkeypatch.setattr(session_module, "_draw_fresh_seed", lambda: next(drawn))
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        config=RuntimeConfig(temperature=1.0, seed=None),
    )

    assert session.resolved_seed == 101
    session.reset()
    assert session.resolved_seed == 202
    session.reseed()
    assert session.resolved_seed == 303


def test_explicit_seed_never_consumes_fresh_entropy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> int:
        raise AssertionError("explicit seeds must not draw fresh entropy")

    monkeypatch.setattr(session_module, "_draw_fresh_seed", unexpected)
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        config=RuntimeConfig(temperature=1.0, seed=77),
    )

    assert session.resolved_seed == 77
    session.reset()
    assert session.resolved_seed == 77


def test_sync_position_appends_reuses_prefix_and_replaces_on_divergence() -> None:
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        config=RuntimeConfig(temperature=0.0, seed=5),
    )
    e2e4 = chess.Move.from_uci("e2e4")
    e7e5 = chess.Move.from_uci("e7e5")
    d2d4 = chess.Move.from_uci("d2d4")
    stream = session.resolved_seed

    opening = session.sync_position(moves=(e2e4,))
    assert opening.replaced is False
    assert opening.reused_prefix_plies == 0
    assert opening.appended_plies == 1

    append = session.sync_position(moves=(e2e4, e7e5))
    assert append.replaced is False
    assert append.reused_prefix_plies == 1
    assert session.move_history == (e2e4, e7e5)

    takeback = session.sync_position(moves=(e2e4,))
    assert takeback.replaced is True
    assert takeback.reused_prefix_plies == 1
    assert session.move_history == (e2e4,)

    divergent = session.sync_position(moves=(d2d4,))
    assert divergent.replaced is True
    assert divergent.reused_prefix_plies == 0
    assert session.move_history == (d2d4,)

    fen = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"
    replaced = session.sync_position(initial_fen=fen, moves=())
    assert replaced.replaced is True
    assert replaced.reused_prefix_plies == 0
    assert session.initial_fen == fen
    # None of the synchronization paths disturbed the seeded stream.
    assert session.resolved_seed == stream


def test_sync_position_rejects_illegal_history_without_mutation() -> None:
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        config=RuntimeConfig(temperature=1.0, seed=8),
    )
    session.sync_position(moves=(chess.Move.from_uci("e2e4"),))
    stream = session.resolved_seed

    with pytest.raises(SessionStateError, match="illegal at ply 0"):
        session.sync_position(moves=(chess.Move.from_uci("e2e5"),))

    assert session.move_history == (chess.Move.from_uci("e2e4"),)
    assert session.resolved_seed == stream


def test_one_session_decides_for_whichever_side_is_to_move() -> None:
    session = GameSession(
        StubRunner(_ranked_logits("e7e5")),
        config=RuntimeConfig(temperature=0.0, seed=11),
    )
    e2e4 = chess.Move.from_uci("e2e4")
    session.sync_position(moves=(e2e4,))
    stream = session.resolved_seed

    # The same session that owns White's history decides for Black.
    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("e7e5")
    assert session.move_history == (e2e4, action.move)
    assert session.resolved_seed == stream

    append = session.sync_position(moves=(e2e4, action.move))
    assert append.replaced is False
    assert append.reused_prefix_plies == 2
    assert session.resolved_seed == stream


def test_reset_validates_history_and_defensively_owns_board_state() -> None:
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
    )
    move = chess.Move.from_uci("d2d4")
    session.reset(moves=(move,))
    exposed = session.board
    exposed.push(chess.Move.from_uci("d7d5"))

    assert session.move_history == (move,)
    with pytest.raises(SessionStateError, match="illegal at ply 0"):
        session.reset(moves=(chess.Move.from_uci("e2e5"),))


def test_terminal_and_observed_move_failures_are_deliberate() -> None:
    runner = StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE))
    session = GameSession(runner)
    with pytest.raises(SessionStateError, match="illegal move"):
        session.apply_move(chess.Move.from_uci("e2e5"))

    terminal = GameSession(runner)
    terminal.reset(
        moves=tuple(
            chess.Move.from_uci(move) for move in ("f2f3", "e7e5", "g2g4", "d8h4")
        )
    )
    assert terminal.is_terminal
    with pytest.raises(SessionStateError, match="terminal"):
        terminal.choose_action()
    with pytest.raises(SessionStateError, match="terminal"):
        terminal.apply_move(chess.Move.null())


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        (torch.zeros(ACTION_VOCABULARY_SIZE - 1), "invalid action-logit shape"),
        (
            torch.full((ACTION_VOCABULARY_SIZE,), float("nan")),
            "non-finite action logits",
        ),
    ],
)
def test_malformed_model_outputs_fail(logits: torch.Tensor, message: str) -> None:
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0),
    )

    with pytest.raises(ActionSelectionError, match=message):
        session.choose_action()
    assert session.move_history == ()


def test_runtime_config_uses_shared_loading_and_enforces_control_bounds(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.toml"
    path.write_text(
        "target_rating = 1600\ntemperature = 0.75\nseed = 17\n",
        encoding="utf-8",
    )

    resolved = load_config(RuntimeConfig, path=path)

    assert resolved.value == RuntimeConfig(
        target_rating=1600,
        temperature=0.75,
        seed=17,
    )
    assert resolved.provenance.source == str(path.resolve())
    # The default seed is unset so ordinary play is not pinned to one stream.
    assert RuntimeConfig().seed is None
    with pytest.raises(ValidationError):
        RuntimeConfig(temperature=-0.01)
    with pytest.raises(ValidationError):
        RuntimeConfig(temperature=float("inf"))
    with pytest.raises(ValidationError):
        RuntimeConfig(seed=-1)


def test_a_decision_reports_the_policy_behind_the_action_it_applied() -> None:
    session = GameSession(
        StubRunner(_ranked_logits("e2e4")),
        config=RuntimeConfig(temperature=0.0),
    )

    decision = session.decide()

    assert isinstance(decision.action, MoveAction)
    assert decision.action.move == chess.Move.from_uci("e2e4")
    # Twenty legal first moves, and the ranked one is the model's preference.
    assert decision.policy.enabled_action_count == 20
    assert decision.policy.selected_rank == 1
    assert decision.policy.preferred_action_id == decision.action.action_id
    assert decision.policy.selected_probability == pytest.approx(
        decision.policy.preferred_probability
    )
    assert 0.0 < decision.policy.selected_probability < 1.0


def test_reported_probabilities_describe_the_model_not_the_temperature() -> None:
    logits = _ranked_logits("e2e4")

    greedy = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0),
    ).decide()
    sampled = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=4),
    ).decide()

    # A temperature change moves which action is drawn, never the model's own
    # confidence in the action it prefers.
    assert greedy.policy.preferred_probability == pytest.approx(
        sampled.policy.preferred_probability
    )
    assert greedy.policy.preferred_action_id == sampled.policy.preferred_action_id


def test_an_off_policy_draw_is_reported_with_its_rank_and_probability() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 10.0
    logits[encode_move(chess.Move.from_uci("d2d4"))] = 9.0
    session = GameSession(
        StubRunner(logits),
        # A high temperature flattens the draw without touching the ranking.
        config=RuntimeConfig(temperature=3.0, seed=1),
    )

    decision = session.decide()

    assert isinstance(decision.action, MoveAction)
    if decision.action.move == chess.Move.from_uci("e2e4"):
        assert decision.policy.selected_rank == 1
    else:
        assert decision.policy.selected_rank > 1
        assert (
            decision.policy.selected_probability < decision.policy.preferred_probability
        )
    assert decision.policy.preferred_action_id == encode_move(
        chess.Move.from_uci("e2e4")
    )


def test_a_resignation_decision_still_reports_its_policy() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[RESIGNATION_ACTION_ID] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=0.0, resignation_enabled=True),
    )

    decision = session.decide()

    assert isinstance(decision.action, ResignationAction)
    assert decision.policy.preferred_action_id == RESIGNATION_ACTION_ID
    # Twenty legal moves plus the enabled resignation action.
    assert decision.policy.enabled_action_count == 21


def test_scoring_an_action_reports_what_deciding_it_would_have_reported() -> None:
    """One selection path, so a re-scored decision is comparable to a live one."""

    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 10.0
    logits[encode_move(chess.Move.from_uci("d2d4"))] = 9.0
    scored = GameSession(StubRunner(logits), config=RuntimeConfig(temperature=0.0))
    decided = GameSession(StubRunner(logits), config=RuntimeConfig(temperature=0.0))

    policy = scored.score_action(encode_move(chess.Move.from_uci("e2e4")))

    assert policy == decided.decide().policy


def test_scoring_reads_the_model_without_deciding_anything() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 10.0
    session = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=9),
    )
    drawn = GameSession(
        StubRunner(logits),
        config=RuntimeConfig(temperature=1.0, seed=9),
    )

    weak = session.score_action(encode_move(chess.Move.from_uci("a2a3")))

    assert weak.selected_rank > 1
    assert weak.preferred_action_id == encode_move(chess.Move.from_uci("e2e4"))
    assert session.move_history == ()
    assert session.board.fen() == chess.STARTING_FEN
    # The stream is untouched, so the next draw is the one it would have made.
    assert _chosen_move(session) == _chosen_move(drawn)


def test_scoring_an_unenabled_action_is_an_error() -> None:
    session = GameSession(
        StubRunner(torch.zeros(ACTION_VOCABULARY_SIZE)),
        config=RuntimeConfig(temperature=0.0),
    )

    with pytest.raises(ActionSelectionError, match="not enabled"):
        session.score_action(RESIGNATION_ACTION_ID)
    with pytest.raises(ActionSelectionError, match="not enabled"):
        session.score_action(encode_move(chess.Move.from_uci("e7e5")))
    with pytest.raises(TypeError):
        session.score_action("e2e4")  # type: ignore[arg-type]


def test_choosing_an_action_stays_the_thin_call_interfaces_use() -> None:
    session = GameSession(
        StubRunner(_ranked_logits("e2e4")),
        config=RuntimeConfig(temperature=0.0),
    )

    action = session.choose_action()

    assert isinstance(action, MoveAction)
    assert action.move == chess.Move.from_uci("e2e4")
    assert session.move_history == (action.move,)


def test_a_split_decision_matches_the_one_the_session_makes_for_itself() -> None:
    """A caller that predicts elsewhere must still decide the same way.

    Batching decisions across games means asking the session for its context
    and handing the logits back, so the split path has to be the ordinary path
    with the prediction lifted out of it.
    """

    logits = _ranked_logits("e2e4")
    whole = GameSession(
        StubRunner(logits), config=RuntimeConfig(temperature=0.5, seed=9)
    )
    split = GameSession(
        StubRunner(logits), config=RuntimeConfig(temperature=0.5, seed=9)
    )

    expected = whole.decide()
    context = split.decision_context()
    decision = split.decide_from_logits(logits.clone())

    assert len(context.plies) == 1
    assert decision == expected
    assert split.move_history == whole.move_history


def test_a_terminal_game_refuses_both_halves_of_a_split_decision() -> None:
    session = GameSession(
        StubRunner(_ranked_logits("e2e4")),
        config=RuntimeConfig(temperature=0.0),
        moves=tuple(
            chess.Move.from_uci(move) for move in ("f2f3", "e7e5", "g2g4", "d8h4")
        ),
    )

    with pytest.raises(SessionStateError, match="terminal game"):
        session.decision_context()
    with pytest.raises(SessionStateError, match="terminal game"):
        session.decide_from_logits(_ranked_logits("e2e4"))


def test_a_played_game_encodes_each_ply_once_however_often_it_is_resent() -> None:
    """Per-decision cost must follow appended plies, not total game length.

    A UCI client resends the whole game before every move, which is the shape
    that used to make encoding quadratic. Counting encodings rather than timing
    them keeps the bound exact and free of machine noise.
    """

    runner = ScriptedRunner()
    session = GameSession(runner, config=RuntimeConfig(temperature=0.0, seed=3))
    board = chess.Board()
    history: list[chess.Move] = []
    decisions = 24

    with _counted_encodings() as counter:
        for _ in range(decisions):
            runner.next_move = _quiet_move(board)
            session.sync_position(moves=tuple(history))
            move = _chosen_move(session)
            board.push(move)
            history.append(move)

    # One encoding per ply the game gained, and nothing for the plies resent.
    assert counter.encodings == decisions
    assert session.move_history == tuple(history)
    assert len(runner.contexts[-1].plies) == decisions


def test_a_reused_prefix_gives_the_model_the_same_context_as_a_rebuild() -> None:
    """Reuse is only correct if the model cannot tell that it happened."""

    runner = ScriptedRunner()
    session = GameSession(runner, config=RuntimeConfig(temperature=0.0, seed=3))
    board = chess.Board()
    history: list[chess.Move] = []

    def play(move: chess.Move | None = None) -> None:
        runner.next_move = move or _quiet_move(board)
        session.sync_position(moves=tuple(history))
        chosen = _chosen_move(session)
        assert runner.contexts[-1] == build_decision_context(
            board,
            tuple(history),
            target_rating=session.config.target_rating,
        )
        assert chosen == runner.next_move
        board.push(runner.next_move)
        history.append(runner.next_move)

    def take_back() -> None:
        board.pop()
        history.pop()

    for _ in range(4):
        play()

    take_back()
    play()  # Resynchronizing onto a shorter history.
    take_back()
    play(_quiet_move(board, skip=1))  # Replacing the last move with another.
    play()

    assert session.move_history == tuple(history)


def _quiet_move(board: chess.Board, *, skip: int = 0) -> chess.Move:
    """Pick a deterministic move that neither ends the game nor repeats.

    ``skip`` selects a later candidate, which is how a test asks for a move
    that diverges from the one ordinary play would have chosen.
    """

    for move in sorted(board.legal_moves, key=lambda move: move.uci()):
        board.push(move)
        playable = not board.is_game_over() and not board.is_repetition(2)
        board.pop()
        if not playable:
            continue
        if skip == 0:
            return move
        skip -= 1
    raise AssertionError("the position has no quiet continuation")


@dataclass
class ScriptedRunner:
    """Force one chosen legal move so a long game can be driven exactly."""

    next_move: chess.Move = chess.Move.null()
    contexts: list[DecisionContext] = field(default_factory=list)

    def predict(self, context: DecisionContext) -> torch.Tensor:
        self.contexts.append(context)
        logits = torch.zeros(ACTION_VOCABULARY_SIZE)
        logits[encode_move(self.next_move)] = 10.0
        return logits


class _EncodingCounter:
    """Count board encodings performed inside one block."""

    def __init__(self) -> None:
        self.encodings = 0


@contextmanager
def _counted_encodings() -> Iterator[_EncodingCounter]:
    counter = _EncodingCounter()
    original = encoding_module._context_for_position

    def counted(**kwargs: object) -> object:
        counter.encodings += 1
        return original(**kwargs)  # type: ignore[arg-type]

    encoding_module._context_for_position = counted  # type: ignore[assignment]
    try:
        yield counter
    finally:
        encoding_module._context_for_position = original


def _chosen_move(session: GameSession) -> chess.Move:
    action = session.choose_action()
    assert isinstance(action, MoveAction)
    return action.move


def _ranked_logits(best: str, *, illegal: str | None = None) -> torch.Tensor:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci(best))] = 10.0
    if illegal is not None:
        logits[encode_move(chess.Move.from_uci(illegal))] = 100.0
    return logits


def test_the_selection_distribution_is_the_one_a_draw_would_sample_from() -> None:
    """An exhaustive walk has to see the same dial the live path samples under."""

    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 2.0
    session = GameSession(StubRunner(logits), config=RuntimeConfig(temperature=1.0))

    enabled, probabilities = session.selection_distribution(logits)

    assert len(enabled) == len(probabilities)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    candidates = logits[torch.tensor(enabled, dtype=torch.long)].to(torch.float64)
    assert torch.allclose(probabilities, torch.softmax(candidates, dim=0))


def test_the_selection_distribution_follows_the_temperature_dial() -> None:
    """At temperature zero the sampling distribution really is a point mass."""

    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 2.0
    session = GameSession(StubRunner(logits), config=RuntimeConfig(temperature=0.0))

    enabled, probabilities = session.selection_distribution(logits)

    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert float(probabilities.max()) == pytest.approx(1.0)
    chosen = enabled[int(torch.argmax(probabilities).item())]
    assert chosen == encode_move(chess.Move.from_uci("e2e4"))


def test_reading_the_selection_distribution_decides_nothing() -> None:
    logits = torch.zeros(ACTION_VOCABULARY_SIZE)
    logits[encode_move(chess.Move.from_uci("e2e4"))] = 2.0
    session = GameSession(
        StubRunner(logits), config=RuntimeConfig(temperature=1.0, seed=5)
    )
    drawn = GameSession(
        StubRunner(logits), config=RuntimeConfig(temperature=1.0, seed=5)
    )

    session.selection_distribution(logits)

    assert session.move_history == ()
    assert session.board.fen() == chess.STARTING_FEN
    assert _chosen_move(session) == _chosen_move(drawn)
