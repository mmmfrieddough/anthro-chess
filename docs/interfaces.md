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

UCI has no engine-side color. Every `go` asks for a move by whichever player
is to move in the synchronized position, so one process serves both sides and
can change sides between or within games.

The protocol-independent runtime has no controlled color either. A session owns
exact game state and decides for the player to move, because exact board state
already identifies that player and the target rating conditions the decision
rather than a player, as recorded in
[`0009-decision-only-rating-conditioning.md`](decisions/0009-decision-only-rating-conditioning.md).
A caller that assigns colors, such as a game loop alternating with a human,
owns that mapping and decides when to ask for a move; the runtime does not
duplicate it. Board state, observed history, the reusable encoded prefix, and
the active random stream are all color-independent, so serving the other side
never reloads the model, restarts a game, or draws a new sampling stream.

Terminal positions return `bestmove 0000`, the UCI null-move representation.
An internal model or inference failure does not masquerade as a terminal
position: the process emits a protocol-safe critical-error diagnostic, records
the detailed failure in the configured log destination, and exits nonzero
without claiming a best move.

The entry point accepts standard application log levels. UCI `debug on|off`
temporarily enables deeper module and command-boundary diagnostics in the same
protocol-safe destination. Logs identify lifecycle events, command names, and
the name and value of every option a GUI sets. Debug logging also emits
versioned JSON game events for accepted position snapshots, engine decisions,
and `ucinewgame` boundaries. Each accepted snapshot records its initial FEN,
complete UCI move list, and resulting FEN, so a replacement, takeback, or game
rooted away from the standard position remains exactly reconstructable. The
events carry a process-session identity and game index; decision events also
carry the resolved runtime settings and sampling seed. The selected checkpoint
is identified by the model runner's lifecycle log.

The events are not written for their own sake: the evaluation layer reads them
back, and `anthro eval decisions` re-scores a played session's decisions against
a checkpoint so a game played in a GUI is analyzed by the same code as a
benchmark rollout. Model distributions stay out of the log because they are
recoverable this way, from the move sequence and the settings that produced it.
Because the checkpoint is a caller declaration rather than something the log
pins down, that command names its own; see the decision-decomposition section of
`docs/evaluation.md`.

This deliberately records accepted chess history rather than raw command
lines. Unknown input, model distributions, corpus records, and other free-form
or sensitive values remain excluded. The events are DEBUG-only, so default
verbosity and volume do not change, and the existing rotating-file policy
bounds sessions that enable them. Snapshot events were chosen over emitting a
second complete-game artifact: the former preserve arbitrary UCI position
replacement as it happens, while generated benchmark games remain owned by the
evaluation artifact layer.

Option values are logged because they are what makes a reported session
reproducible: the engine's options are bounded scalars a GUI chose
deliberately, and the resolved sampling seed is already recorded in full.
Withholding the settings that produced a game while recording the seed that
replays it is not a coherent boundary. A future option carrying free-form text
would change that judgment and should be excluded when it is introduced.

This first path is untimed and move-only. It ignores unsupported `go` fields
rather than treating them as model inputs. `searchmoves`, pondering, clock
fields, `movetime`, depth/node/mate limits, asynchronous cancellation, and
portable non-move game actions remain outside the implemented scope.

`go infinite` is honored to the extent the protocol requires: the move is still
chosen synchronously, but the response is withheld until `stop` rather than
sent immediately. There is no deeper search behind the wait. A `stop` with no
held response stays a no-op, because an unpaired `bestmove` would be a worse
protocol violation than ignoring the command, and a held response is discarded
by `ucinewgame`.

A malformed move token in `position` rejects the whole command and preserves
the previously synchronized position. Skipping the token would shift every
later move one ply earlier and silently desynchronize the engine from the GUI.

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
option name UCI_LimitStrength type check default false
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

## Non-Move Game Actions

The core Anthro runtime supports resignation and draw claims as learned game
actions, each behind its own runtime setting and both off by default.

Standard UCI carries neither. Every `go` command is expected to end with a
`bestmove` response, and the protocol has no engine-to-GUI command for
resigning, claiming, offering, or accepting a draw. This was a deliberate
narrowing relative to the older Chess Engine Communication Protocol, which does
give the engine `resign` and `offer draw`; UCI moved those decisions to the GUI.
Extension proposals exist but are not portable.

Portable UCI mode therefore rejects a configuration that enables either action
rather than disabling it silently, and refuses to answer a terminal action as
though it were a move. A host-specific extension or a native Anthro interface
may expose them directly.

The cost of disabling differs between the two. Suppressing resignation changes
observable behavior, because the bot plays on in positions where a human would
have stopped. Suppressing draw claims usually costs nothing, because UCI hosts
adjudicate repetition and the fifty-move rule themselves.

## Native Interfaces

The native `anthro` CLI currently provides the package's lightweight command
surface, installation smoke check, and PGN sample-data preparation route. Its
handlers remain thin and call importable package APIs as training, evaluation,
play, and runtime capabilities are implemented.

The CLI also reports the machine itself. Corpora and runs live outside every
worktree beneath configured roots, so a checkout cannot say what this machine
holds, and from inside one an unconfigured root looks exactly like an empty
one. `anthro machine` answers that in a single place — which roots are set,
what is beneath them, and how the default model selection resolves — and exits
nonzero when the configuration is itself the defect rather than the artifacts
being absent. `anthro model select` maintains that default selection record.
Failure text belongs to the same concern: a command that needs a root and
cannot resolve one names the variable and what it would have to hold, instead
of reporting a missing artifact.

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
