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
from typing import Annotated, Any

import chess
from pydantic import Field, StrictBool, StrictInt

from anthro_chess.chess import RESIGNATION_ACTION_ID, decode_move
from anthro_chess.config import ConfigModel
from anthro_chess.evaluation.games.players import (
    DecisionRequest,
    GamePlayer,
    PlayerError,
    PlayerSeat,
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


def generate_games(
    first: GamePlayer,
    second: GamePlayer,
    positions: Sequence[StartPosition],
    *,
    config: GenerationConfig | None = None,
) -> Iterator[GameRecord]:
    """Play every planned game and yield one record each.

    Records are yielded rather than returned so a long suite can be written to
    the detail tier incrementally instead of accumulating in memory.
    """

    resolved = config or GenerationConfig()
    if not positions:
        raise GenerationError("a generation suite needs at least one position")
    for plan in _plan_games(first, second, positions, resolved):
        yield _play_game(plan, resolved)


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


def _play_game(plan: _GamePlan, config: GenerationConfig) -> GameRecord:
    position = plan.position
    board = position.board()
    state = _GameState(board=board, action_ids=list(position.prefix_action_ids))
    seats: dict[PlayerSlot, PlayerSeat] = {
        "white": plan.white.seat(seed=plan.white_seed),
        "black": plan.black.seat(seed=plan.black_seed),
    }
    try:
        outcome = _play_out(state, seats, position, config)
    finally:
        for seat in seats.values():
            seat.close()
    record = build_game_record(
        initial_position=position.initial_position,
        prefix_plies=len(position.prefix_action_ids),
        action_ids=state.action_ids,
        white=plan.white.seat_record(seed=plan.white_seed),
        black=plan.black.seat_record(seed=plan.black_seed),
        seed=plan.seed,
        decisions=state.decisions,
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


def _play_out(
    state: _GameState,
    seats: dict[PlayerSlot, PlayerSeat],
    position: StartPosition,
    config: GenerationConfig,
) -> GameOutcome:
    """Decide plies until the rules or the ply limit end the game.

    A prefix that is already terminal yields a game with no decisions rather
    than an error. A perturbed or truncated position source can produce one,
    and recording it honestly keeps it visible in the suite's distribution
    instead of failing the whole run.
    """

    while True:
        finished = _rule_outcome(state.board, config)
        if finished is not None:
            return finished
        if len(state.decisions) >= config.maximum_generated_plies:
            return GameOutcome(
                result="*",
                termination=GameTermination.PLY_LIMIT,
                adjudicated=True,
            )
        slot: PlayerSlot = "white" if state.board.turn == chess.WHITE else "black"
        request = DecisionRequest(
            board=state.board.copy(stack=True),
            initial_position=position.initial_position,
            ply_index=len(state.action_ids),
        )
        decision = seats[slot].decide(request)
        state.decisions.append(
            DecisionRecord(
                ply_index=len(state.action_ids),
                slot=slot,
                action_id=decision.action_id,
                policy=decision.policy,
            )
        )
        state.action_ids.append(decision.action_id)
        if decision.action_id == RESIGNATION_ACTION_ID:
            return GameOutcome(
                result="0-1" if state.board.turn == chess.WHITE else "1-0",
                termination=GameTermination.RESIGNATION,
                adjudicated=False,
            )
        move = decode_move(decision.action_id)
        if move not in state.board.legal_moves:
            raise PlayerError(
                f"seat {slot} returned illegal move {move.uci()} in the "
                "position it was asked about"
            )
        state.board.push(move)


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
