"""Generated-game machinery shared by every benchmark that plays games.

Three layers, useful independently:

- a versioned game-record format written by both the harness here and by
  reconstruction from live runtime logs, so a manually played game and a
  benchmark rollout are read by one analysis pass;
- a generation harness that is two player configurations plus a position
  source, which covers self-play, mixed-rating ladders, external-engine
  anchors, and random-opponent robustness arms without a mode flag;
- an analysis layer that turns retained records into distribution features, so
  a new feature is recomputed over games already on disk.

A shared *corpus* of generated games is deliberately not the goal. The settings
differ in ways that block reuse, so the saving is in the machinery rather than
in the games.
"""

from anthro_chess.evaluation.games.analysis import (
    GAME_ANALYSIS_VERSION,
    GameDistribution,
    GameFeatures,
    RepetitionDiagnostics,
    analyze_game,
    analyze_games,
    summarize_games,
)
from anthro_chess.evaluation.games.harness import (
    GENERATION_VERSION,
    GenerationConfig,
    GenerationError,
    StartPosition,
    generate_games,
    prefix_positions,
    standard_positions,
)
from anthro_chess.evaluation.games.players import (
    DecisionRequest,
    ExternalEngine,
    ExternalEnginePlayer,
    GamePlayer,
    ModelPlayer,
    PlayerError,
    PlayerSeat,
    RandomPlayer,
    SeatDecision,
    UciEngineProcess,
)
from anthro_chess.evaluation.games.records import (
    GAME_RECORD_VERSION,
    ClockState,
    DecisionPolicy,
    DecisionRecord,
    GameOutcome,
    GameRecord,
    GameRecordError,
    GameResult,
    GameTermination,
    PlayerSlot,
    SeatRecord,
    build_game_record,
    parse_game_records,
    read_game_records,
    termination_from_outcome,
    write_game_records,
)

__all__ = [
    "GAME_ANALYSIS_VERSION",
    "GAME_RECORD_VERSION",
    "GENERATION_VERSION",
    "ClockState",
    "DecisionPolicy",
    "DecisionRecord",
    "DecisionRequest",
    "ExternalEngine",
    "ExternalEnginePlayer",
    "GameDistribution",
    "GameFeatures",
    "GameOutcome",
    "GamePlayer",
    "GameRecord",
    "GameRecordError",
    "GameResult",
    "GameTermination",
    "GenerationConfig",
    "GenerationError",
    "ModelPlayer",
    "PlayerError",
    "PlayerSeat",
    "PlayerSlot",
    "RandomPlayer",
    "RepetitionDiagnostics",
    "SeatDecision",
    "SeatRecord",
    "StartPosition",
    "UciEngineProcess",
    "analyze_game",
    "analyze_games",
    "build_game_record",
    "generate_games",
    "parse_game_records",
    "prefix_positions",
    "read_game_records",
    "standard_positions",
    "summarize_games",
    "termination_from_outcome",
    "write_game_records",
]
