# Interfaces

Anthro Chess should expose the model through practical interfaces without
making any outside protocol part of the model itself.

The internal boundary is the Anthro runtime API. It accepts exact game state,
target rating, temperature, optional preference settings, and optional timing
context. It returns a sampled game action and, when timing is enabled, a
sampled move time.

Outside interfaces should translate between their protocol and that runtime
API.

The current `anthro_chess.runtime.GameSession` is the implemented untimed
boundary. It accepts typed chess moves and model-runner outputs, not protocol
text, and returns typed move or resignation actions after updating its owned
game state. The installed `anthro-uci` process is the first implemented
frontend over that boundary.

## UCI

Universal Chess Interface, or UCI, should be the default compatibility
interface for local chess GUIs and engine tools. The protocol reference is the
April 2006 UCI specification:
<https://backscattering.de/chess/uci/>.

UCI is a text protocol over a child process's standard input and standard
output. A chess GUI starts the Anthro Chess executable, sends commands such as
`uci`, `isready`, `position`, `setoption`, and `go`, and waits for responses
such as `uciok`, `readyok`, and `bestmove`.

Anthro Chess should support UCI by direct invocation: the UCI executable should
load the Anthro runtime and model in the same process.

Standard output in UCI mode is reserved for UCI protocol messages. The current
entry point writes timestamped application diagnostics to a bounded rotating
local file. `--log-file` selects an explicit destination, while
`ANTHRO_CHESS_LOG_ROOT` can relocate the default application-log directory.
If the file cannot be initialized, logging falls back to standard error without
writing non-protocol text to standard output.

The current package installs `anthro-uci` as a dedicated console script. It
loads strict UCI configuration at process startup, completes the `uci`
handshake without loading model weights, and initializes the compatible
retained checkpoint when the GUI first synchronizes with `isready`. Model
selection may be explicit or resolve beneath `ANTHRO_CHESS_RUN_ROOT`, so the
process does not depend on a repository working directory. This is an
installed Python entry point, not a frozen standalone binary.

The operator-facing CPU smoke, persistent configuration, GUI setup, acceptance
procedure, and current product limitations are documented in
[`playable-uci.md`](playable-uci.md).

The initial implementation supports `uci`, `debug`, `isready`, `setoption`,
`ucinewgame`, `position`, synchronous `go`, `stop`, and `quit`. Position
synchronization accepts `startpos` or a complete FEN plus move history and
commits only after the complete history validates. It exposes
`UCI_LimitStrength`, `UCI_Elo`, `Anthro Temperature`, and `Anthro Seed`; their
exact bounds, scaling, and the random-seed sentinel live in the UCI
configuration module. Disabling `UCI_LimitStrength` selects the code-owned
maximum conditioning rating, not a claim of calibrated playing strength, and
is the protocol-default state. Unknown commands and tokens are ignored where
the remaining command can be parsed safely.

Terminal positions return `bestmove 0000`, the UCI null-move representation.
An internal model or inference failure does not masquerade as a terminal
position: the process emits a protocol-safe critical-error diagnostic, records
the detailed failure in the configured log destination, and exits nonzero
without claiming a best move.

The entry point accepts standard application log levels. UCI `debug on|off`
temporarily enables deeper module and command-boundary diagnostics in the same
protocol-safe destination. Logs identify lifecycle events and command names,
but do not copy raw commands, complete game histories, model distributions, or
other high-volume or sensitive values.

This first path is untimed and move-only. It accepts `stop` safely for the
short synchronous inference path and ignores unsupported `go` fields rather
than treating them as model inputs. Analysis search, `searchmoves`, pondering,
clock fields, `movetime`, `infinite`, depth/node/mate limits, asynchronous
cancellation, and portable resignation remain outside the implemented scope.

## UCI Scope

The UCI layer should:

- identify the engine with `id` and `uciok` after `uci`;
- advertise configurable options after `uci`;
- respond to `isready` with `readyok`;
- apply `setoption` updates to the current engine process;
- reset game-local state on `ucinewgame`;
- reconstruct the current game from `position`;
- use `go` clock fields as timing context when timing is enabled;
- return a legal move with `bestmove`.

The UCI layer should not be the model's native representation. The model should
not learn UCI commands or raw protocol text.

UCI callers may send a complete `position` before every `go`. The interface
should validate the supplied state exactly while reusing the current canonical
prefix when it is an append-only update. A new FEN, takeback, or divergent move
list must still work through an atomic replacement path. Position
synchronization must preserve the loaded model and other process-lifetime
resources, invalidate only incompatible cached history, and must not reset the
sampling generator.

## UCI Command Handling

Anthro Chess should implement the standard UCI commands that make sense for a
direct learned policy:

- `uci`: enter UCI mode, identify the engine, advertise options, and send
  `uciok`;
- `debug on|off`: toggle extra diagnostics in the application log without
  adding arbitrary stdout output;
- `isready`: finish required initialization and respond `readyok`;
- `setoption`: update process-local configuration;
- `ucinewgame`: clear game-local runtime state;
- `position`: set the board from `startpos` or `fen` plus UCI move history;
- `go`: sample an Anthro action for the current position;
- `stop`: stop waiting or generation as soon as practical and return
  `bestmove`;
- `quit`: exit cleanly.

`ponderhit` can be accepted as a no-op until Anthro Chess intentionally supports
pondering. Legacy registration commands can be ignored unless a future packaged
distribution needs them.

`go` fields should be interpreted according to what Anthro can use:

- `wtime`, `btime`, `winc`, and `binc` provide clock context when timing is
  enabled;
- `movetime` may be treated as a caller-imposed maximum or fixed response
  window rather than a model-training feature;
- `searchmoves` should restrict the legal move mask to the provided legal
  moves;
- search-depth fields such as `depth`, `nodes`, and `mate` do not naturally map
  to Anthro's learned-policy runtime and can be ignored unless needed for tool
  compatibility;
- `infinite` should wait for `stop` if used in analysis contexts.

## UCI Options

UCI options are declared by the engine and set by the GUI. They are not a
general bidirectional settings system. The engine can advertise default values,
minimums, maximums, and enumerated choices, but there is no standard message
for the engine to push an updated option value back into the GUI.

Anthro Chess should use a normal project config file or command-line arguments
for initial defaults. UCI options should override those defaults for the
running engine process when a GUI sends `setoption`.

Useful UCI options include:

- `UCI_LimitStrength`, mapped to whether target rating control is active;
- `UCI_Elo`, mapped internally to target rating;
- custom temperature options;
- a custom seed option that supports fresh per-game randomness by default and
  explicit reproducible sampling when selected;
- custom timing toggles or timing-style options;
- custom soft preference controls.

Examples:

```text
option name UCI_LimitStrength type check default true
option name UCI_Elo type spin default 1500 min 400 max 2500
option name Anthro Temperature type spin default 100 min 0 max 300
option name Anthro Seed type spin default -1 min -1 max 2147483647
option name Anthro Aggression type spin default 0 min -100 max 100
option name Anthro Sicilian type spin default 0 min -100 max 100
option name Anthro Time Conservation type spin default 0 min -100 max 100
```

Custom preference controls can use UCI `spin`, `check`, `combo`, or `string`
types, but GUI presentation quality will vary. Some GUIs may show numeric
fields rather than polished sliders. A native Anthro interface can provide a
better control surface if UCI GUI options are too limited.

## Resignation

The core Anthro runtime may support resignation as a learned game action.

Standard UCI does not provide a portable engine-to-GUI resignation response.
Every `go` command is expected to end with a `bestmove` response. UCI mode
should therefore disable resignation by default or handle it only through a
host-specific extension. Native Anthro interfaces may expose resignation
directly.

## Native Interfaces

The native `anthro` CLI currently provides the package's lightweight command
surface, installation smoke check, and PGN sample-data preparation route. Its
handlers remain thin and call importable package APIs as training, evaluation,
play, and runtime capabilities are implemented.

A future native web UI may talk directly to the Anthro runtime API. Native
interfaces can expose richer controls, show current runtime state, and
represent non-move game actions such as resignation without being limited by
UCI.

Native interfaces should share the same config schema as UCI mode where
possible, but UCI GUI settings should not be expected to stay synchronized with
settings changed elsewhere after the engine process has started.

Seed behavior must be consistent across interfaces. Temperature zero is
deterministic regardless of seed. At nonzero temperature, interactive
interfaces use fresh game randomness by default, while an explicit seed
reproduces a run. A position synchronization is not a new sampling run and must
not reseed it. See
[`0010-separate-position-sync-from-randomness.md`](decisions/0010-separate-position-sync-from-randomness.md).
