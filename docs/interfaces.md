# Interfaces

Anthro Chess should expose the model through practical interfaces without
making any outside protocol part of the model itself.

The internal boundary is the Anthro runtime API. It accepts exact game state,
target rating, temperature, optional preference settings, and optional timing
context. It returns a sampled game action and, when timing is enabled, a
sampled move time.

Outside interfaces should translate between their protocol and that runtime
API.

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

Standard output in UCI mode must be reserved for UCI protocol messages. Normal
application logs, model-loading messages, progress output, and diagnostics
should go to files or standard error.

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

## UCI Command Handling

Anthro Chess should implement the standard UCI commands that make sense for a
direct learned policy:

- `uci`: enter UCI mode, identify the engine, advertise options, and send
  `uciok`;
- `debug on|off`: toggle extra protocol-safe diagnostics, using `info string`
  or file logs rather than arbitrary stdout output;
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
- custom timing toggles or timing-style options;
- custom soft preference controls.

Examples:

```text
option name UCI_LimitStrength type check default true
option name UCI_Elo type spin default 1500 min 400 max 2500
option name Anthro Temperature type spin default 100 min 0 max 300
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
surface and installation smoke check. Its handlers should remain thin and call
importable package APIs as data, training, evaluation, play, and runtime
capabilities are implemented.

A future native web UI may talk directly to the Anthro runtime API. Native
interfaces can expose richer controls, show current runtime state, and
represent non-move game actions such as resignation without being limited by
UCI.

Native interfaces should share the same config schema as UCI mode where
possible, but UCI GUI settings should not be expected to stay synchronized with
settings changed elsewhere after the engine process has started.
