# Playing Anthro Chess Through UCI

Anthro Chess can be launched as an installed UCI engine with a compatible
retained checkpoint. The engine runs the model and decision runtime in the same
process; it does not require a server or a repository-relative Python command.

The current proof is intentionally narrow. It supports untimed games, exact
position replacement, legal move selection, new-game reset, target-rating and
temperature options, and terminal positions. The selected development
checkpoint is weak and is not a published model release.

## Select The Engine And Checkpoint

Create the locked environment and retain its absolute UCI executable path:

```console
uv sync --locked
export ANTHRO_UCI_EXECUTABLE="$PWD/.venv/bin/anthro-uci"
```

Set `ANTHRO_CHESS_RUN_ROOT` to the machine-local directory containing complete
training runs. The current Playable Proof checkpoint is retained in
`decision-conditioned-rating-proof-v3`:

```console
export ANTHRO_CHESS_RUN_ROOT="/absolute/path/to/anthro-chess/runs"
```

The run directory must contain its `run.json`, checkpoint directory, and
compatibility metadata. Do not copy out only the weight file.

For a persistent setup, save a strict configuration file outside the
repository. The GUI launcher reads it from `gui.toml` in the state directory
described under [Connect A Chess GUI](#connect-a-chess-gui); any other location
works for direct invocation:

```toml
[model]
checkpoint_path = "/absolute/run/checkpoints/step-00002000.pt"
device = "cpu"
```

Set only what the machine cannot infer. Every `[runtime]` setting has a
code-owned default, and each one a GUI can reach is an advertised UCI option
that overrides the file for the running process. Restating a default in the
file gains nothing and invites the belief that the file is what the engine is
actually using, so prefer the GUI option and leave the file alone.

Restating `runtime.temperature` is the trap worth naming: it sets the
advertised `Anthro Temperature` default, so a file pinning it to zero makes
every game from a position identical until the GUI raises the option. That is
sampling behavior, not a model property, and it is easy to mistake for one.

`runtime.target_rating` is the one setting that does not take effect on its
own. It seeds the advertised `UCI_Elo` default, but following UCI convention
strength limiting is off until the GUI enables `UCI_LimitStrength`, and until
then the engine conditions on the code-owned maximum rating.

Omitting `seed` uses fresh per-game randomness; set an explicit non-negative
`seed` only to reproduce a game.

An absolute `model.checkpoint_path` makes GUI startup independent of inherited
environment variables while still requiring the complete retained run around
the checkpoint. Relative `model.run_path` selections are also supported and
resolve beneath `ANTHRO_CHESS_RUN_ROOT`.

## CPU Command-Line Smoke

Run the installed executable from a directory unrelated to the repository:

```console
cd /tmp
printf 'uci\nisready\nposition startpos\ngo\nquit\n' |
  "$ANTHRO_UCI_EXECUTABLE" \
    --set 'model.run_path="decision-conditioned-rating-proof-v3"' \
    --set 'model.checkpoint="step-00002000.pt"' \
    --set 'model.device="cpu"'
```

A successful smoke prints engine identification, `uciok`, `readyok`, and one
legal `bestmove`. Model initialization diagnostics go to the configured
application log, not standard output. This smoke proves the direct CPU path and
checkpoint compatibility; it does not assert playing strength.

## Connect A Chess GUI

Configure the GUI's UCI engine command with the absolute path to
`scripts/anthro-uci-gui` in the main checkout, and no arguments. The exact
fields vary by GUI, but the GUI must launch that script directly and
communicate through its standard input and output. Do not point it at
`src/anthro_chess/interfaces/uci.py`, at a path inside `.venv`, or at a
worktree.

The launcher is a committed shell entry point, so it survives `uv sync`, branch
switches, and worktree removal. It decides at launch which checkout serves the
protocol, reports that decision and any failure on standard error, and leaves
standard output to the engine alone.

Its shared engine configuration lives outside every checkout, next to the
target pointer, and is the file described under
[Select The Engine And Checkpoint](#select-the-engine-and-checkpoint). Override
its location with `ANTHRO_CHESS_GUI_CONFIG`, the state directory with
`ANTHRO_CHESS_GUI_ROOT`, and log verbosity with `ANTHRO_CHESS_GUI_LOG_LEVEL`.
Those variables are for manual runs from a terminal: a GUI started from a
desktop launcher does not inherit a login shell environment, which is why the
target is a file rather than an environment variable.

## Point The GUI At A Branch

The GUI is configured once. Which checkout it serves is a separate, switchable
decision, so a change can be tried in a real GUI without touching GUI settings.

```console
scripts/anthro-gui-target            # print the current target
scripts/anthro-gui-target .          # serve this checkout
scripts/anthro-gui-target --clear    # fall back to the launcher's checkout
```

Run it from the checkout to be served, after that checkout's environment is
initialized with `uv sync`. Restart the engine in the GUI to pick up the change;
most GUIs reload it when a new game starts.

With no pointer, the launcher serves the checkout it lives in, so the default is
the main checkout and no configuration is needed for ordinary play.

If a pointed-at worktree is removed, the launcher fails with a readable message
naming the missing path instead of starting a stale or partial engine. Run
`scripts/anthro-gui-target --clear` to recover.

After the GUI completes the UCI handshake, it can set:

- `UCI_LimitStrength` to enable or disable the selected target rating;
- `UCI_Elo` to choose the target rating while strength limiting is enabled;
- `Anthro Temperature` to control sampling independently, scaled by 100;
- `Anthro Seed` to select fresh per-game randomness or a reproducible game.

Position synchronization keeps the loaded model and the active random stream
alive, so at nonzero temperature ordinary interactive games vary by default and
a repeated position no longer collapses to the same continuation. `Anthro Seed`
selects a reproducible game when set to an explicit value and returns to fresh
per-game randomness at its sentinel; temperature zero stays deterministic
regardless of seed. `ucinewgame` starts a fresh game and stream without
reloading the model. The exact seed range and sentinel are owned by the UCI
configuration module. See
[`0010-separate-position-sync-from-randomness.md`](decisions/0010-separate-position-sync-from-randomness.md).

For final Playable Proof acceptance:

1. Start an untimed standard-chess game and confirm Anthro returns legal moves.
2. Finish or deliberately adjudicate the game.
3. Start a new game without restarting the engine configuration and confirm
   Anthro moves again, including a game where Anthro takes the other color.
4. Quit the GUI and confirm the engine process exits.
5. Record the GUI name and version, operating system, completed-game outcome,
   and successful new-game reset in issue #35.

## Current Boundaries

The initial UCI process is move-only and synchronous. It does not support
analysis search, pondering, `searchmoves`, clock-aware timing, `movetime`,
infinite analysis, depth or node search, hard cancellation of an in-flight
model forward pass, or portable resignation. Unsupported `go` fields are
ignored rather than used as model inputs.

`UCI_Elo` selects learned rating conditioning; it is not yet calibrated proof
of the engine's playing strength. Disabling `UCI_LimitStrength` selects the
maximum supported conditioning rating and does not turn Anthro into a
conventional strongest-line engine.

The selected proof checkpoint can produce plausible local play while still
showing weak separation across configured ratings and frequent deterministic
repetition in generated games. Those are model-quality and rollout-evaluation
findings, not claims established by the UCI integration test. Generated-game
benchmarks should measure them across seeds, colors, temperatures, and frozen
human prefixes rather than drawing conclusions from one GUI game.

Detailed application diagnostics are written to a bounded rotating log. See
[`interfaces.md`](interfaces.md) for the protocol boundary and logging
behavior.
