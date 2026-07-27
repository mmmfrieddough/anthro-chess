"""The one record format every generated or reconstructed game is written in.

Six planned benchmarks need whole games, and two different producers write
them: the generation harness in this package, and reconstruction from live
runtime logs. One format for both is what lets a manually played game and a
benchmark rollout be read by one analysis pass instead of two.

A record carries what its consumers cannot recover by replaying the moves. The
move sequence and the initial position reconstruct the board, so nothing
derivable from them is stored. Per-decision policy quantities, the resolved
seat configurations, and the resolved seed are not derivable, so they are.

Clock state is absent rather than synthesized. The model has no time head yet,
and a zero or an invented remaining time would be indistinguishable from a
measured one to every later reader.

The record version is this format's own, independent of the result envelope in
``anthro_chess.evaluation.results``. Games are bulk diagnostics that live in
the machine-local detail tier; only the metrics computed from them reach the
committed summary tier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal

import chess
from pydantic import BaseModel, ConfigDict, Field, model_validator

from anthro_chess.chess import (
    ACTION_VOCABULARY_SIZE,
    MOVE_ACTION_COUNT,
    RESIGNATION_ACTION_ID,
    decode_move,
)
from anthro_chess.evaluation.results.records import (
    CheckpointReference,
    DetailReference,
    canonical_json,
)
from anthro_chess.evaluation.results.store import DetailStore
from anthro_chess.runtime import SelectionPolicy

GAME_RECORD_VERSION = 1

#: Result strings match the normalized corpus, so a generated distribution and
#: a human reference distribution are counted over the same vocabulary.
GameResult = Literal["1-0", "0-1", "1/2-1/2", "*"]

PlayerSlot = Literal["white", "black"]

ActionId = Annotated[int, Field(ge=0, lt=ACTION_VOCABULARY_SIZE)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class GameRecordError(ValueError):
    """Raised when a game record is malformed or cannot be persisted."""


class GameTermination(StrEnum):
    """Why a game ended.

    The rule-derived cases come from the chess layer and are exact. The two
    remaining cases are the harness's own: a learned resignation, and the ply
    limit that stops an unfinished game. Both are distinguished from a rule
    ending because a suite that adjudicates half its games is reporting
    something different from one that plays them out.
    """

    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    SEVENTYFIVE_MOVES = "seventyfive_moves"
    FIVEFOLD_REPETITION = "fivefold_repetition"
    FIFTY_MOVES = "fifty_moves"
    THREEFOLD_REPETITION = "threefold_repetition"
    RESIGNATION = "resignation"
    PLY_LIMIT = "ply_limit"

    @property
    def claimed(self) -> bool:
        """Return whether this ending needed a player to claim a draw."""

        return self in _CLAIMED_TERMINATIONS


_CLAIMED_TERMINATIONS = frozenset(
    {GameTermination.FIFTY_MOVES, GameTermination.THREEFOLD_REPETITION}
)

_LIBRARY_TERMINATIONS = {
    chess.Termination.CHECKMATE: GameTermination.CHECKMATE,
    chess.Termination.STALEMATE: GameTermination.STALEMATE,
    chess.Termination.INSUFFICIENT_MATERIAL: GameTermination.INSUFFICIENT_MATERIAL,
    chess.Termination.SEVENTYFIVE_MOVES: GameTermination.SEVENTYFIVE_MOVES,
    chess.Termination.FIVEFOLD_REPETITION: GameTermination.FIVEFOLD_REPETITION,
    chess.Termination.FIFTY_MOVES: GameTermination.FIFTY_MOVES,
    chess.Termination.THREEFOLD_REPETITION: GameTermination.THREEFOLD_REPETITION,
}


def termination_from_outcome(outcome: chess.Outcome) -> GameTermination:
    """Map a chess-layer outcome onto this format's termination vocabulary."""

    try:
        return _LIBRARY_TERMINATIONS[outcome.termination]
    except KeyError:
        raise GameRecordError(
            f"unsupported game termination: {outcome.termination.name}"
        ) from None


class GameRecordModel(BaseModel):
    """Base for the immutable, code-owned parts of a game record."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ClockState(GameRecordModel):
    """Clock state at one decision, present only when timing exists."""

    remaining_ms: int = Field(ge=0)
    increment_ms: int | None = Field(default=None, ge=0)


class DecisionPolicy(GameRecordModel):
    """What the policy said about one selected action.

    The probabilities are the model's own distribution over enabled actions,
    not the tempered one the draw used, so they describe the model rather than
    the dial. The seat's temperature is recorded beside them.
    """

    enabled_action_count: int = Field(ge=1)
    selected_probability: Probability
    selected_rank: int = Field(ge=1)
    preferred_action_id: ActionId
    preferred_probability: Probability

    @classmethod
    def from_selection(cls, policy: SelectionPolicy) -> DecisionPolicy:
        """Return the stored form of one runtime selection policy."""

        return cls(
            enabled_action_count=policy.enabled_action_count,
            selected_probability=policy.selected_probability,
            selected_rank=policy.selected_rank,
            preferred_action_id=policy.preferred_action_id,
            preferred_probability=policy.preferred_probability,
        )

    @model_validator(mode="after")
    def _validate_rank(self) -> DecisionPolicy:
        if self.selected_rank > self.enabled_action_count:
            raise ValueError("a selected action cannot rank below the enabled count")
        if self.selected_probability > self.preferred_probability:
            raise ValueError("no action can be likelier than the preferred one")
        return self


class DecisionRecord(GameRecordModel):
    """One decision a seat made, at the ply it was made."""

    ply_index: int = Field(ge=0)
    slot: PlayerSlot
    action_id: ActionId
    policy: DecisionPolicy | None = None
    clock: ClockState | None = None


class SeatRecord(GameRecordModel):
    """The resolved configuration one seat played a game under.

    ``configuration`` is the seat's own resolved settings rather than a digest,
    because a rating ladder or a temperature grid is read by inspecting exactly
    these values across records. Records are machine-local, so they are not
    bound by the committed tier's size budget.
    """

    kind: Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9-]*$")]
    label: Annotated[str, Field(min_length=1)]
    seed: int | None = Field(default=None, ge=0)
    configuration: dict[str, Any] = Field(default_factory=dict)
    checkpoint: CheckpointReference | None = None


class GameOutcome(GameRecordModel):
    """How a game ended, and whether the harness ended it."""

    result: GameResult
    termination: GameTermination
    adjudicated: bool

    @model_validator(mode="after")
    def _validate_adjudication(self) -> GameOutcome:
        if self.termination is GameTermination.PLY_LIMIT:
            if not self.adjudicated:
                raise ValueError("a ply-limit ending is always adjudicated")
            if self.result != "*":
                raise ValueError("a ply-limit ending has no result to report")
        elif self.result == "*":
            raise ValueError(f"{self.termination.value} must report a decided result")
        return self


class GameRecord(GameRecordModel):
    """One whole game, in the shape every consumer of generated play reads."""

    record_version: int = Field(ge=1)
    game_id: Annotated[str, Field(pattern=r"^[a-f0-9]{16}$")]
    initial_position: Annotated[str, Field(min_length=1)]
    #: Plies replayed from a frozen source rather than chosen by a seat. Zero
    #: for a game generated from its initial position.
    prefix_plies: int = Field(ge=0)
    action_ids: tuple[ActionId, ...]
    white: SeatRecord
    black: SeatRecord
    seed: int = Field(ge=0)
    decisions: tuple[DecisionRecord, ...]
    outcome: GameOutcome
    #: Provenance of an injected prefix: which pool game it came from, and the
    #: label the position source gave it.
    source_game_id: int | None = Field(default=None, ge=0)
    position_label: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_history(self) -> GameRecord:
        if self.prefix_plies > len(self.action_ids):
            raise ValueError("a prefix cannot be longer than the game it starts")
        resignations = [
            index
            for index, action_id in enumerate(self.action_ids)
            if action_id == RESIGNATION_ACTION_ID
        ]
        if resignations and resignations != [len(self.action_ids) - 1]:
            raise ValueError("a resignation can only be a game's final action")
        expected_plies = list(range(self.prefix_plies, len(self.action_ids)))
        if [decision.ply_index for decision in self.decisions] != expected_plies:
            raise ValueError("decisions must cover every ply past the prefix, in order")
        board = _root_board(self.initial_position)
        for decision in self.decisions:
            if self.action_ids[decision.ply_index] != decision.action_id:
                raise ValueError(
                    f"decision at ply {decision.ply_index} disagrees with the "
                    "recorded action"
                )
            expected_slot: PlayerSlot = (
                "white" if _slot_at(board, decision.ply_index) else "black"
            )
            if decision.slot != expected_slot:
                raise ValueError(
                    f"decision at ply {decision.ply_index} names the wrong seat"
                )
        _replay(board, self.action_ids)
        return self

    @property
    def move_action_ids(self) -> tuple[int, ...]:
        """Return the move actions only, dropping a terminal resignation."""

        return tuple(
            action_id for action_id in self.action_ids if action_id < MOVE_ACTION_COUNT
        )

    @property
    def ply_count(self) -> int:
        """Return how many moves the game contains."""

        return len(self.move_action_ids)

    @property
    def generated_plies(self) -> int:
        """Return how many plies a seat actually chose."""

        return len(self.decisions)

    def seat(self, slot: PlayerSlot) -> SeatRecord:
        """Return the seat record for one color."""

        return self.white if slot == "white" else self.black

    def board(self) -> chess.Board:
        """Replay the game and return its final position."""

        board = _root_board(self.initial_position)
        _replay(board, self.action_ids)
        return board

    def as_record(self) -> dict[str, Any]:
        """Return the JSON-compatible record written to the detail tier."""

        return self.model_dump(mode="json")


def build_game_record(
    *,
    initial_position: str,
    prefix_plies: int,
    action_ids: Sequence[int],
    white: SeatRecord,
    black: SeatRecord,
    seed: int,
    decisions: Sequence[DecisionRecord],
    outcome: GameOutcome,
    source_game_id: int | None = None,
    position_label: str | None = None,
) -> GameRecord:
    """Assemble a validated record with a content-derived identity.

    The identity is derived from the record itself, so the same game generated
    twice from the same seed is the same record rather than two entries that
    only a reader can tell apart.
    """

    record = GameRecord(
        record_version=GAME_RECORD_VERSION,
        game_id="0" * 16,
        initial_position=initial_position,
        prefix_plies=prefix_plies,
        action_ids=tuple(action_ids),
        white=white,
        black=black,
        seed=seed,
        decisions=tuple(decisions),
        outcome=outcome,
        source_game_id=source_game_id,
        position_label=position_label,
    )
    payload = record.model_dump(mode="json")
    payload.pop("game_id", None)
    identity = sha256(canonical_json(payload)).hexdigest()[:16]
    return record.model_copy(update={"game_id": identity})


def write_game_records(
    store: DetailStore,
    relative_path: str,
    records: Iterable[GameRecord],
    *,
    description: str | None = None,
) -> DetailReference:
    """Write games to the machine-local detail tier and reference them.

    Generated games are bulk diagnostics by definition, so they never reach the
    committed summary tier. A benchmark records its scalar measurements there
    and points at this payload.
    """

    payload = {
        "version": GAME_RECORD_VERSION,
        "games": [record.as_record() for record in records],
    }
    return store.write(relative_path, payload, description=description)


def read_game_records(
    store: DetailStore,
    reference: DetailReference,
) -> tuple[GameRecord, ...]:
    """Read back a referenced payload of games, verifying its digest."""

    return parse_game_records(store.read(reference))


def parse_game_records(payload: Any) -> tuple[GameRecord, ...]:
    """Validate a persisted payload of games against this build's format."""

    if not isinstance(payload, Mapping):
        raise GameRecordError("a game payload must be an object")
    version = payload.get("version")
    if version != GAME_RECORD_VERSION:
        raise GameRecordError(
            f"game payload uses record version {version!r}; this build "
            f"understands {GAME_RECORD_VERSION}"
        )
    games = payload.get("games")
    if not isinstance(games, Sequence) or isinstance(games, str | bytes):
        raise GameRecordError("a game payload must carry a list of games")
    return tuple(GameRecord.model_validate(game) for game in games)


def _root_board(initial_position: str) -> chess.Board:
    try:
        return chess.Board(initial_position)
    except ValueError as error:
        raise ValueError(f"invalid initial position: {error}") from error


def _slot_at(root: chess.Board, ply_index: int) -> chess.Color:
    """Return which color moves at one ply of a game rooted at ``root``."""

    return root.turn if ply_index % 2 == 0 else not root.turn


def _replay(board: chess.Board, action_ids: Sequence[int]) -> None:
    """Apply a recorded action sequence to ``board``, rejecting illegal play.

    A terminal resignation ends the replay: it is an action rather than a move,
    so it leaves the position where the last move left it.
    """

    for ply_index, action_id in enumerate(action_ids):
        if action_id == RESIGNATION_ACTION_ID:
            return
        move = decode_move(action_id)
        if move not in board.legal_moves:
            raise ValueError(
                f"recorded action at ply {ply_index} ({move.uci()}) is illegal "
                "in the position it is played from"
            )
        board.push(move)
