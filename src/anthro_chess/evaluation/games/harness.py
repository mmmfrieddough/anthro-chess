"""Two player configurations plus a position source, played out into records.

That is the whole abstraction, and it is deliberately the smallest one that
covers every planned consumer. Self-play puts one configuration in both seats,
a rating ladder puts two ratings in them, an engine anchor puts an external
process in one, and a robustness rollout puts a uniform-random opponent there.
None of those is a mode this module knows about.

Nothing here waits in wall-clock time or draws from an uncontrolled source. A
suite is identified by one base seed; every game's seed and every seat's stream
are derived from it, so a single game can be reproduced on its own from the
seed its record carries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import islice
from typing import Annotated, Any

import chess
from pydantic import Field, StrictBool, StrictInt

from anthro_chess.chess import RESIGNATION_ACTION_ID, decode_move
from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.games.players import (
    BatchingGamePlayer,
    DecisionRequest,
    GamePlayer,
    PlayerError,
    PlayerSeat,
    SeatDecision,
)
from anthro_chess.evaluation.games.records import (
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameResult,
    GameTermination,
    PlayerSlot,
    build_game_record,
    termination_from_outcome,
)
from anthro_chess.evaluation.results.records import canonical_json

GENERATION_VERSION = 1

#: Seeds stay positive and inside a signed 64-bit integer so they are safe to
#: log, to record, and to hand to a seeded generator.
_SEED_BITS = 63

logger = logging.getLogger(__name__)


class GenerationError(ValueError):
    """Raised when a game cannot be generated from the requested inputs."""


@dataclass(frozen=True)
class StartPosition:
    """One root a game is generated from.

    A frozen human prefix is a root plus the moves already played, not a
    separate mode: the harness replays the prefix and then asks the seats to
    continue, so a prefix continuation and a game from the standard position
    differ only in how many plies were injected.
    """

    initial_position: str = chess.STARTING_FEN
    prefix_action_ids: tuple[int, ...] = ()
    label: str | None = None
    source_game_id: int | None = None

    def board(self) -> chess.Board:
        """Return the position the seats start deciding from."""

        try:
            board = chess.Board(self.initial_position)
        except ValueError as error:
            raise GenerationError(f"invalid initial position: {error}") from error
        for ply_index, action_id in enumerate(self.prefix_action_ids):
            if action_id == RESIGNATION_ACTION_ID:
                raise GenerationError("a prefix cannot contain a resignation")
            move = decode_move(action_id)
            if move not in board.legal_moves:
                raise GenerationError(
                    f"prefix move {ply_index} ({move.uci()}) is illegal in the "
                    "position it is played from"
                )
            board.push(move)
        return board


def standard_positions(
    count: int = 1,
    *,
    label: str | None = None,
) -> tuple[StartPosition, ...]:
    """Return roots at the standard starting position."""

    if count < 1:
        raise GenerationError("a position source needs at least one position")
    return tuple(StartPosition(label=label) for _ in range(count))


def prefix_positions(
    games: Iterable[tuple[int, Sequence[int]]],
    *,
    plies: int,
    initial_position: str = chess.STARTING_FEN,
) -> tuple[StartPosition, ...]:
    """Return roots continuing frozen human games after ``plies`` moves.

    Games shorter than the requested prefix are dropped rather than truncated
    to whatever they have. A shorter prefix is a different measurement, and
    silently mixing depths would make the continuation length an uncontrolled
    variable.
    """

    if plies < 1:
        raise GenerationError("a prefix continuation needs at least one ply")
    positions = []
    for game_id, action_ids in games:
        if len(action_ids) < plies:
            logger.debug("Skipping game %s: shorter than the prefix", game_id)
            continue
        positions.append(
            StartPosition(
                initial_position=initial_position,
                prefix_action_ids=tuple(action_ids[:plies]),
                label=f"prefix-{plies}",
                source_game_id=game_id,
            )
        )
    if not positions:
        raise GenerationError("no source game is long enough for the requested prefix")
    return tuple(positions)


class GenerationConfig(ConfigModel):
    """Code-owned settings for one generation suite."""

    seed: Annotated[StrictInt, Field(ge=0)] = 0
    #: Games per position for each color assignment, so the realized count is
    #: this many times two when colors are swapped.
    games_per_position: Annotated[StrictInt, Field(ge=1)] = 1
    #: How many plies the seats may add past the prefix before the game is
    #: adjudicated unfinished.
    maximum_generated_plies: Annotated[StrictInt, Field(ge=1)] = 300
    #: Play both color assignments. A model that is stronger with white than
    #: with black would otherwise show up as a property of the matchup.
    swap_colors: StrictBool = True
    #: Whether the harness claims a draw the rules make claimable. Off by
    #: default: the model has no draw-claim action, so claiming for it would
    #: report the harness's policy as the model's behavior. Games still end on
    #: their own through the fivefold and seventy-five-move rules.
    claim_draws: StrictBool = False
    #: How many games are played at once. Concurrent games are advanced in
    #: lock step so one player's pending decisions can be resolved together;
    #: one keeps the sequential path. Games never observe each other, so this
    #: is a throughput setting rather than a measurement setting.
    concurrency: Annotated[StrictInt, Field(ge=1)] = 1


@dataclass
class _GamePlan:
    """One game's resolved seats and seed, before anything is played."""

    position: StartPosition
    white: GamePlayer
    black: GamePlayer
    seed: int
    white_seed: int
    black_seed: int


@dataclass
class _GameState:
    """Mutable state accumulated while one game is played out."""

    board: chess.Board
    action_ids: list[int] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)


class _GameRun:
    """One game in flight: its plan, its open seats, and how far it has gone."""

    def __init__(self, *, plan: _GamePlan) -> None:
        self.plan = plan
        self.state = _GameState(
            board=plan.position.board(),
            action_ids=list(plan.position.prefix_action_ids),
        )
        self.seats: dict[PlayerSlot, PlayerSeat] = {
            "white": plan.white.seat(seed=plan.white_seed),
            "black": plan.black.seat(seed=plan.black_seed),
        }
        self.outcome: GameOutcome | None = None

    def close(self) -> None:
        """Release both seats' per-game state."""

        for seat in self.seats.values():
            seat.close()


@dataclass(frozen=True)
class _PendingDecision:
    """One game waiting on one seat to choose an action."""

    run: _GameRun
    slot: PlayerSlot
    request: DecisionRequest

    @property
    def player(self) -> GamePlayer:
        """Return the player configuration occupying the deciding seat."""

        return self.run.plan.white if self.slot == "white" else self.run.plan.black

    @property
    def seat(self) -> PlayerSeat:
        """Return the open seat this decision belongs to."""

        return self.run.seats[self.slot]


def generate_games(
    first: GamePlayer,
    second: GamePlayer,
    positions: Sequence[StartPosition],
    *,
    config: GenerationConfig | None = None,
) -> Iterator[GameRecord]:
    """Play every planned game and yield one record each.

    Records are yielded rather than returned so a long suite can be written to
    the detail tier incrementally instead of accumulating in memory. Concurrent
    games are played in waves, so a suite accumulates one wave of records at a
    time rather than one game's.
    """

    resolved = config or GenerationConfig()
    if not positions:
        raise GenerationError("a generation suite needs at least one position")
    width = _wave_width(first, second, resolved)
    plans = _plan_games(first, second, positions, resolved)
    while wave := tuple(islice(plans, width)):
        yield from _play_wave(wave, resolved)


def _wave_width(
    first: GamePlayer,
    second: GamePlayer,
    config: GenerationConfig,
) -> int:
    """Return how many games may be in flight at once for this pairing.

    A player whose seats cannot overlap, such as one external engine process
    serving every game, forces the suite back to one game at a time. Refusing
    to overlap is cheaper than a wrong measurement, and the pairing is known
    before anything is played.
    """

    if config.concurrency == 1:
        return 1
    if first.supports_concurrent_games and second.supports_concurrent_games:
        return config.concurrency
    logger.info(
        "Playing one game at a time: a seat in this pairing cannot overlap games",
    )
    return 1


def _plan_games(
    first: GamePlayer,
    second: GamePlayer,
    positions: Sequence[StartPosition],
    config: GenerationConfig,
) -> Iterator[_GamePlan]:
    assignments = (False, True) if config.swap_colors else (False,)
    for position_index, position in enumerate(positions):
        for swapped in assignments:
            white, black = (second, first) if swapped else (first, second)
            for game_index in range(config.games_per_position):
                seed = _derive_seed(
                    config.seed,
                    position_index,
                    position.label,
                    position.source_game_id,
                    int(swapped),
                    game_index,
                )
                yield _GamePlan(
                    position=position,
                    white=white,
                    black=black,
                    seed=seed,
                    white_seed=_derive_seed(seed, "white"),
                    black_seed=_derive_seed(seed, "black"),
                )


def _play_wave(
    plans: Sequence[_GamePlan],
    config: GenerationConfig,
) -> tuple[GameRecord, ...]:
    """Play one wave of games in lock step and return their records in order.

    Every game in the wave advances one ply per round, so the decisions the
    round produces for a single player configuration can be resolved together.
    Games never observe each other: each keeps its own board and each seat its
    own random stream, and a game that ends simply stops contributing
    decisions while the rest of the wave continues. A wave of one is the
    sequential path.
    """

    runs = [_GameRun(plan=plan) for plan in plans]
    try:
        while True:
            pending = tuple(
                filter(None, (_pending_decision(run, config) for run in runs))
            )
            if not pending:
                break
            for player, group in _grouped_by_player(pending):
                _resolve(player, group)
    finally:
        for run in runs:
            run.close()
    return tuple(_finished_record(run) for run in runs)


def _pending_decision(
    run: _GameRun,
    config: GenerationConfig,
) -> _PendingDecision | None:
    """Return the decision this game is waiting on, or nothing once it ended.

    A prefix that is already terminal yields a game with no decisions rather
    than an error. A perturbed or truncated position source can produce one,
    and recording it honestly keeps it visible in the suite's distribution
    instead of failing the whole run.
    """

    if run.outcome is not None:
        return None
    finished = _rule_outcome(run.state.board, config)
    if finished is not None:
        run.outcome = finished
        return None
    if len(run.state.decisions) >= config.maximum_generated_plies:
        run.outcome = GameOutcome(
            result="*",
            termination=GameTermination.PLY_LIMIT,
            adjudicated=True,
        )
        return None
    slot: PlayerSlot = "white" if run.state.board.turn == chess.WHITE else "black"
    return _PendingDecision(
        run=run,
        slot=slot,
        request=DecisionRequest(
            board=run.state.board.copy(stack=True),
            initial_position=run.plan.position.initial_position,
            ply_index=len(run.state.action_ids),
        ),
    )


def _grouped_by_player(
    pending: Sequence[_PendingDecision],
) -> tuple[tuple[GamePlayer, tuple[_PendingDecision, ...]], ...]:
    """Group one round's pending decisions by the player configuration to ask.

    Grouping is by identity rather than by equality, because one player object
    in both seats is exactly the self-play case that should share a pass, while
    two players configured alike are still two configurations.
    """

    groups: list[tuple[GamePlayer, list[_PendingDecision]]] = []
    for item in pending:
        player = item.player
        for known, members in groups:
            if known is player:
                members.append(item)
                break
        else:
            groups.append((player, [item]))
    return tuple((player, tuple(members)) for player, members in groups)


def _resolve(player: GamePlayer, group: Sequence[_PendingDecision]) -> None:
    """Choose and apply one action for every decision pending on one player."""

    if isinstance(player, BatchingGamePlayer) and len(group) > 1:
        decisions = player.decide_batch(
            tuple((item.seat, item.request) for item in group)
        )
        if len(decisions) != len(group):
            raise PlayerError("a batched player returned the wrong decision count")
    else:
        decisions = tuple(item.seat.decide(item.request) for item in group)
    for item, decision in zip(group, decisions, strict=True):
        _apply(item, decision)


def _apply(pending: _PendingDecision, decision: SeatDecision) -> None:
    """Record one chosen action and advance or end the game it belongs to."""

    run = pending.run
    state = run.state
    state.decisions.append(
        DecisionRecord(
            ply_index=len(state.action_ids),
            slot=pending.slot,
            action_id=decision.action_id,
            policy=decision.policy,
        )
    )
    state.action_ids.append(decision.action_id)
    if decision.action_id == RESIGNATION_ACTION_ID:
        run.outcome = GameOutcome(
            result="0-1" if state.board.turn == chess.WHITE else "1-0",
            termination=GameTermination.RESIGNATION,
            adjudicated=False,
        )
        return
    move = decode_move(decision.action_id)
    if move not in state.board.legal_moves:
        raise PlayerError(
            f"seat {pending.slot} returned illegal move {move.uci()} in the "
            "position it was asked about"
        )
    state.board.push(move)


def _finished_record(run: _GameRun) -> GameRecord:
    """Build the record of one game the wave played to an outcome."""

    plan = run.plan
    position = plan.position
    outcome = run.outcome
    if outcome is None:  # pragma: no cover - the wave only returns on outcomes
        raise GenerationError("cannot record a game that has not ended")
    record = build_game_record(
        initial_position=position.initial_position,
        prefix_plies=len(position.prefix_action_ids),
        action_ids=run.state.action_ids,
        white=plan.white.seat_record(seed=plan.white_seed),
        black=plan.black.seat_record(seed=plan.black_seed),
        seed=plan.seed,
        decisions=run.state.decisions,
        outcome=outcome,
        source_game_id=position.source_game_id,
        position_label=position.label,
    )
    logger.debug(
        "Generated game %s: %s plies, %s by %s",
        record.game_id,
        record.ply_count,
        outcome.result,
        outcome.termination.value,
    )
    return record


def _rule_outcome(board: chess.Board, config: GenerationConfig) -> GameOutcome | None:
    outcome = board.outcome(claim_draw=config.claim_draws)
    if outcome is None:
        return None
    termination = termination_from_outcome(outcome)
    result: GameResult = "1/2-1/2"
    if outcome.winner is not None:
        result = "1-0" if outcome.winner == chess.WHITE else "0-1"
    # A claimed draw is the harness exercising a rule the seats have no action
    # for, so it counts as adjudicated even though the rules allowed it.
    return GameOutcome(
        result=result,
        termination=termination,
        adjudicated=termination.claimed,
    )


def _derive_seed(*parts: Any) -> int:
    """Derive one reproducible seed from a game's identifying coordinates.

    Hashing rather than counting a running stream means a seed is a pure
    function of those coordinates. A single game can then be regenerated on its
    own from the seed its record carries, without replaying the suite that
    produced it in order.
    """

    payload = canonical_json(list(parts))
    return int.from_bytes(sha256(payload).digest()[:8], "big") >> (64 - _SEED_BITS)
