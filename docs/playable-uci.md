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
repository:

```toml
[model]
checkpoint_path = "/absolute/run/checkpoints/step-00002000.pt"
device = "cpu"

[runtime]
target_rating = 1500
temperature = 0.0
seed = 0
resignation_enabled = false
```

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

Configure the GUI's UCI engine command with:

- executable: the absolute path in `ANTHRO_UCI_EXECUTABLE`;
- arguments: `--config /absolute/path/to/anthro-uci.toml`.

The exact fields vary by GUI, but the GUI must launch that executable directly
and communicate through its standard input and output. Do not point it at
`src/anthro_chess/interfaces/uci.py`.

After the GUI completes the UCI handshake, it can set:

- `UCI_LimitStrength` to enable or disable the selected target rating;
- `UCI_Elo` to choose the target rating while strength limiting is enabled;
- `Anthro Temperature` to control sampling independently, scaled by 100.

The current proof uses a fixed configured seed. Its position-replacement path
also resets that seed whenever the GUI synchronizes the game, so even
nonzero-temperature play is reproducible for the same position and settings.
This is a known proof-stage limitation rather than the intended interactive
behavior. The intended runtime keeps randomness alive across position updates,
uses fresh per-game randomness for ordinary play, and exposes an explicit seed
for reproducible games and benchmarks. See
[`0010-separate-position-sync-from-randomness.md`](decisions/0010-separate-position-sync-from-randomness.md).

For final Playable Proof acceptance:

1. Start an untimed standard-chess game and confirm Anthro returns legal moves.
2. Finish or deliberately adjudicate the game.
3. Start a new game without restarting the engine configuration and confirm
   Anthro moves again.
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
