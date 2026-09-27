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

Set `ANTHRO_CHESS_RUN_ROOT` to the machine-local directory holding complete
training runs, as described in `CONTRIBUTING.md`:

```console
export ANTHRO_CHESS_RUN_ROOT="/absolute/path/to/anthro-chess/runs"
```

The run directory must contain its `run.json`, checkpoint directory, and
compatibility metadata. Do not copy out only the weight file.

Which run to play is a machine-local decision rather than a documented one.
The engine validates a checkpoint against the current action vocabulary and
encoding and refuses an incompatible one, so a run trained before either was
last bumped is no longer playable and naming one here would go stale on the
next bump. The run root's default model selection record answers the question
instead: when it names a run, the engine needs no model arguments at all, and
otherwise every command below takes an explicit selection.

For a persistent setup, save a strict configuration file outside the
repository and pass it with `--config`:

```toml
[model]
checkpoint_path = "/absolute/run/checkpoints/<checkpoint>.pt"
device = "cpu"
```

Set only what the machine cannot infer. Every `[runtime]` setting has a
code-owned default, and each one a client can set is an advertised UCI option
that overrides the file for the running process. Restating a default in the
file gains nothing and invites the belief that the file is what the engine is
actually using, so prefer the UCI option and leave the file alone.

Restating `runtime.temperature` is the trap worth naming: it sets the
advertised `Anthro Temperature` default, so a file pinning it to zero makes
every game from a position identical until the client raises the option. That is
sampling behavior, not a model property, and it is easy to mistake for one.

`runtime.target_rating` is the one setting that does not take effect on its
own. It seeds the advertised `UCI_Elo` default, but following UCI convention
strength limiting is off until the client enables `UCI_LimitStrength`, and until
then the engine conditions on the code-owned maximum rating.

Omitting `seed` uses fresh per-game randomness; set an explicit non-negative
`seed` only to reproduce a game.

An absolute `model.checkpoint_path` makes startup independent of inherited
environment variables while still requiring the complete retained run around
the checkpoint. Relative `model.run_path` selections are also supported and
resolve beneath `ANTHRO_CHESS_RUN_ROOT`.

## CPU Command-Line Smoke

Run the installed executable from a directory unrelated to the repository. With
a default model selection recorded in the run root, the engine needs no model
arguments:

```console
cd /tmp
printf 'uci\nisready\nposition startpos\ngo\nquit\n' |
  "$ANTHRO_UCI_EXECUTABLE" --set 'model.device="cpu"'
```

To smoke a particular run instead, name it. A relative run path resolves
beneath `ANTHRO_CHESS_RUN_ROOT`, and omitting `model.checkpoint` takes the
run's latest:

```console
printf 'uci\nisready\nposition startpos\ngo\nquit\n' |
  "$ANTHRO_UCI_EXECUTABLE" \
    --set 'model.run_path="<run>"' \
    --set 'model.device="cpu"'
```

A successful smoke prints engine identification, `uciok`, `readyok`, and one
legal `bestmove`. Model initialization diagnostics go to the configured
application log, not standard output. This smoke proves the direct CPU path and
checkpoint compatibility; it does not assert playing strength. A checkpoint the
current build cannot load fails here with a compatibility message, which is the
cheapest way to find out that a run predates a vocabulary or encoding bump.

## Engine Options

After the UCI handshake, any client, including the Lichess preview below, can
set:

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

## Preview A Change On Lichess

To try a change by playing it, put the checkout's engine on Lichess as a bot and
challenge it from lichess.org or the Lichess app, from any device.
`scripts/lichess-preview.py` does this for the checkout it is run from, so a
worktree previews itself:

```console
uv run scripts/lichess-preview.py start [--run <run>] [--checkpoint <file>] [--rating <elo>] [--temperature <t>]
uv run scripts/lichess-preview.py status
uv run scripts/lichess-preview.py stop
```

Without `--run` it serves the machine's default model selection. Without
`--rating` strength limiting stays off, as described under
[Engine Options](#engine-options), and without `--temperature` the engine's
default applies. `start` resolves the selection to one checkpoint before the bot
goes online, and `status` reports the checkout, the checkpoint, the options, and
any game in progress.

The bot accepts casual standard games at any time control, from one Lichess
account only. A machine has one bot account, so one preview runs at a time:
`start` refuses while another is running and names the checkout it serves.
Stopping mid-game abandons the game, so `stop` refuses while one is in progress
unless given `--force`.

The engine logs every decision at debug level, with the seed that drew it, to
the application log [`interfaces.md`](interfaces.md) describes, so a previewed
game can be replayed and analyzed afterwards.

### One-Time Setup

Each machine that serves previews needs:

1. A Lichess account for the bot, created fresh. Lichess converts only an
   account that has never played a game, and the conversion is permanent. Bot
   accounts are labelled as bots, and their profiles and games are public.
2. A personal API token for that account with the `bot:play` scope.
3. The account converted to a bot, once, with that token:

   ```console
   curl -X POST -H "Authorization: Bearer <token>" https://lichess.org/api/bot/account/upgrade
   ```

4. The token and the one Lichess username allowed to challenge the bot, as
   `LICHESS_BOT_TOKEN` and `ANTHRO_CHESS_PREVIEW_OPPONENT`. Set them in the
   environment, or as `NAME=value` lines in
   `~/.config/anthro-chess/lichess-preview.env` (beneath `XDG_CONFIG_HOME` when
   that is set), which every checkout and worktree on the machine shares. The
   environment wins where both are set. Keep the file readable only by its
   owner.

On first use the script fetches lichess-bot, the program that connects an
engine to the Lichess bot API, at the revision the script pins. It lives beneath
`~/.local/state/anthro-chess/lichess-preview/` (or `XDG_STATE_HOME`) with the
preview's log and state. A missing value, a token without the scope, or an
account that is not yet a bot fails `start` with a message pointing back here.

## Current Boundaries

The initial UCI process is move-only and synchronous. It does not support
analysis search, pondering, `searchmoves`, clock-aware timing, `movetime`,
infinite analysis, depth or node search, hard cancellation of an in-flight
model forward pass, or portable non-move game actions. Unsupported `go` fields
are ignored rather than used as model inputs.

`UCI_Elo` selects learned rating conditioning; it is not yet calibrated proof
of the engine's playing strength. Disabling `UCI_LimitStrength` selects the
maximum supported conditioning rating and does not turn Anthro into a
conventional strongest-line engine.

The selected proof checkpoint can produce plausible local play while still
showing weak separation across configured ratings and frequent deterministic
repetition in generated games. Those are model-quality and rollout-evaluation
findings, not claims established by the UCI integration test. Generated-game
benchmarks should measure them across seeds, colors, temperatures, and frozen
human prefixes rather than drawing conclusions from one played game.

Detailed application diagnostics are written to a bounded rotating log. See
[`interfaces.md`](interfaces.md) for the protocol boundary and logging
behavior.
